// Jarvis daemon: держит Python-мост живым и будит ассистента только по имени.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"
)

var wakeRe = regexp.MustCompile(`^\s*(?:джарвис|jarvis)(?:[\s,.:!\-]+|$)(.*)`)

type event struct {
	Type  string `json:"type"`
	Text  string `json:"text"`
	Reply string `json:"reply"`
	Tag   string `json:"tag"`
	Done  bool   `json:"done"`
	Error string `json:"error"`
	Model string `json:"model"`
}

func main() {
	python := flag.String("python", "", "Путь к python (по умолчанию .venv/bin/python)")
	workdir := flag.String("workdir", "", "Корень репозитория Jarvis")
	lang := flag.String("lang", "ru", "Язык STT/ответов")
	model := flag.String("model", "qwen:14b", "Модель Ollama")
	mute := flag.Bool("mute", false, "Без озвучки")
	flag.Parse()
	if flag.NArg() > 0 {
		fmt.Fprintln(os.Stderr, "подсказка: демон не принимает позиционные аргументы")
		fmt.Fprintln(os.Stderr, "пример: go run .")
		fmt.Fprintln(os.Stderr, "опции: -lang ru -model qwen:14b -mute")
		fmt.Fprintln(os.Stderr, "лишние аргументы:", strings.Join(flag.Args(), " "))
	}

	root, err := resolveWorkdir(*workdir)
	if err != nil {
		fatal(err)
	}
	py, err := resolvePython(root, *python)
	if err != nil {
		fatal(err)
	}

	fmt.Println("Jarvis daemon")
	fmt.Println("workdir:", root)
	fmt.Println("python:", py)
	fmt.Println("wake word: Джарвис / Jarvis")
	fmt.Println("пример: «Джарвис, два плюс два» — ок; «как дела» — игнор")
	fmt.Println("«Джарвис, ты свободен» — выход")
	fmt.Println(strings.Repeat("-", 40))

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	args := []string{"-m", "voice", "bridge", "--lang", *lang, "--ollama-model", *model}
	if *mute {
		args = append(args, "--mute")
	}

	for {
		exit, runErr := runSession(root, py, args, sigCh)
		if runErr != nil {
			fmt.Fprintln(os.Stderr, "сессия:", runErr)
		}
		if exit {
			fmt.Println("Демон остановлен.")
			return
		}
		fmt.Println("Перезапуск моста через 1с…")
		select {
		case <-sigCh:
			fmt.Println("Демон остановлен.")
			return
		case <-time.After(time.Second):
		}
	}
}

func runSession(root, py string, args []string, sigCh <-chan os.Signal) (bool, error) {
	cmd := exec.Command(py, args...)
	cmd.Dir = root
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return true, err
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return true, err
	}
	if err := cmd.Start(); err != nil {
		return true, err
	}

	waitErr := make(chan error, 1)
	go func() { waitErr <- cmd.Wait() }()

	events := make(chan event, 16)
	readErr := make(chan error, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			var ev event
			if err := json.Unmarshal(scanner.Bytes(), &ev); err != nil {
				fmt.Fprintln(os.Stderr, "bad event:", scanner.Text())
				continue
			}
			events <- ev
		}
		if err := scanner.Err(); err != nil {
			readErr <- err
			return
		}
		readErr <- io.EOF
	}()

	busy := false
	for {
		select {
		case <-sigCh:
			_ = writeJSON(stdin, map[string]string{"type": "shutdown"})
			_ = cmd.Process.Signal(os.Interrupt)
			select {
			case <-waitErr:
			case <-time.After(3 * time.Second):
				_ = cmd.Process.Kill()
				<-waitErr
			}
			return true, nil
		case err := <-waitErr:
			if err != nil {
				return false, err
			}
			return false, nil
		case err := <-readErr:
			select {
			case <-waitErr:
			case <-time.After(2 * time.Second):
				_ = cmd.Process.Kill()
				<-waitErr
			}
			if err == io.EOF {
				return false, nil
			}
			return false, err
		case ev := <-events:
			switch ev.Type {
			case "ready":
				fmt.Println("мост готов:", ev.Model)
			case "heard":
				fmt.Println("слышу:", ev.Text)
				if busy {
					continue
				}
				command, ok := extractCommand(ev.Text)
				if !ok {
					fmt.Println("  → без имени, пропуск")
					continue
				}
				if command == "" {
					command = "привет"
				}
				fmt.Println("  → команда:", command)
				busy = true
				if err := writeJSON(stdin, map[string]string{"type": "act", "text": command}); err != nil {
					busy = false
					return false, err
				}
			case "reply":
				fmt.Printf("ответ (%s): %s\n", ev.Tag, ev.Reply)
				busy = false
				if ev.Done {
					_ = writeJSON(stdin, map[string]string{"type": "shutdown"})
					select {
					case <-waitErr:
					case <-time.After(3 * time.Second):
						_ = cmd.Process.Kill()
						<-waitErr
					}
					return true, nil
				}
			case "bye":
				select {
				case <-waitErr:
				case <-time.After(3 * time.Second):
					_ = cmd.Process.Kill()
					<-waitErr
				}
				return true, nil
			case "error":
				fmt.Fprintln(os.Stderr, "bridge error:", ev.Error)
				busy = false
			}
		}
	}
}

func extractCommand(text string) (string, bool) {
	normalized := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(text), "ё", "е"))
	match := wakeRe.FindStringSubmatch(normalized)
	if match == nil {
		return "", false
	}
	return strings.TrimSpace(match[1]), true
}

func writeJSON(w io.Writer, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = w.Write(append(data, '\n'))
	return err
}

func resolveWorkdir(explicit string) (string, error) {
	if explicit != "" {
		return filepath.Abs(explicit)
	}
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if filepath.Base(wd) == "daemon" {
		return filepath.Abs(filepath.Join(wd, ".."))
	}
	if _, err := os.Stat(filepath.Join(wd, "voice")); err == nil {
		return wd, nil
	}
	return filepath.Abs(filepath.Join(wd, ".."))
}

func resolvePython(root, explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	candidate := filepath.Join(root, ".venv", "bin", "python")
	if _, err := os.Stat(candidate); err == nil {
		return candidate, nil
	}
	path, err := exec.LookPath("python3")
	if err != nil {
		return "", fmt.Errorf("python не найден: создайте .venv или укажите -python")
	}
	return path, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ошибка:", err)
	os.Exit(1)
}
