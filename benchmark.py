"""Compare original and physically pruned models on fixed CPU workloads."""
import json
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

CASES = {
    "math": [
        ("What is 17 times 19?", "323"), ("What is 28 times 16?", "448"),
        ("What is 144 divided by 12?", "12"), ("What is 15 percent of 240?", "36"),
        ("What is the next term in 2, 6, 18, 54?", "162"),
        ("If x + 7 = 21, what is x?", "14"),
    ],
    "code": [
        ("In Python, what does len([1, 2, 3]) return?", "3"),
        ("In Python, what does 'abc'[::-1] return?", "cba"),
        ("In Python, what does sum(range(5)) return?", "10"),
        ("In Python, what does sorted([3, 1, 2])[0] return?", "1"),
        ("In JavaScript, what does typeof null return?", "object"),
        ("In Python, what does 17 % 5 return?", "2"),
    ],
    "facts": [
        ("What is the capital of Kenya?", "Nairobi"),
        ("What is the chemical symbol for sodium?", "Na"),
        ("What planet is closest to the Sun?", "Mercury"),
        ("What organ pumps blood around the human body?", "heart"),
        ("What is the largest ocean on Earth?", "Pacific"),
        ("What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ],
    "multilingual": [
        ("¿Cuál es la capital de Francia? Responde en español.", "París"),
        ("Wie heißt die Hauptstadt von Deutschland? Antworte auf Deutsch.", "Berlin"),
        ("Qual è la capitale d'Italia? Rispondi in italiano.", "Roma"),
        ("भारत की राजधानी क्या है? हिंदी में उत्तर दें।", "नई दिल्ली"),
        ("日本の首都はどこですか？日本語で答えてください。", "東京"),
        ("Quelle est la capitale de la Belgique ? Réponds en français.", "Bruxelles"),
    ],
}


def query(route, data=None):
    request = Request("http://127.0.0.1:8083/" + route,
                      data=None if data is None else json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def generate(prompt, tokens):
    return query("v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0, "seed": 42, "max_tokens": tokens, "cache_prompt": False,
    })


def main():
    root = Path("/var/tmp/qwen-pruning")
    model, = Path("/var/tmp").glob("qwen-models.*/Qwen3.5-4B-Q4_K_M.gguf")
    binary = Path.home() / "llama.cpp/build/bin/llama-server"
    with (root / "benchmark.jsonl").open("x") as output:
        for variant, path in [("original", model), ("norm", root / "norm.gguf"), ("random", root / "random.gguf")]:
            with (root / f"{variant}.log").open("w") as log, subprocess.Popen([
                str(binary), "--model", str(path), "--gpu-layers", "0", "--threads", "4",
                "--ctx-size", "512", "--parallel", "1", "--host", "127.0.0.1", "--port", "8083",
            ], stdout=log, stderr=log) as server:
                try:
                    for _ in range(60):
                        if server.poll() is not None:
                            raise RuntimeError(f"{variant} failed to start")
                        try:
                            if query("health")["status"] == "ok":
                                break
                        except URLError:
                            time.sleep(1)
                    else:
                        raise RuntimeError(f"{variant} readiness timeout")

                    for category, cases in CASES.items():
                        for prompt, expected in cases:
                            result = generate("Answer with only the shortest correct answer. " + prompt, 12)
                            answer = result["choices"][0]["message"]["content"]
                            print(json.dumps(dict(variant=variant, category=category, prompt=prompt,
                                                  expected=expected, answer=answer), ensure_ascii=False),
                                  file=output, flush=True)
                    for _ in range(3):
                        result = generate("Write a 200-word introduction to the history of computing.", 96)
                        rss = next(line.split()[1] for line in Path(f"/proc/{server.pid}/status")
                                   .read_text().splitlines() if line.startswith("VmRSS:"))
                        print(json.dumps(dict(variant=variant, category="performance",
                                              rss_kib=int(rss), bytes=path.stat().st_size,
                                              timings=result["timings"])), file=output, flush=True)
                finally:
                    if server.poll() is None:
                        server.terminate()


if __name__ == "__main__":
    main()
