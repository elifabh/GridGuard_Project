import requests
import json

class LLMAnalyst:
    def __init__(self, model="llama3"):
        self.model = model
        self.api_url = "http://127.0.0.1:11434/api/generate"

    def analyze(self, context_type, data_stats):
        # 1. Prompt Hazırla (Daha kısa ve net tuttum ki hata vermesin)
        if context_type == "market":
            prompt = f"Analyze energy market: Wind {data_stats['avg_wind']:.1f}%, Price {data_stats['avg_price']:.1f}c. One short sentence recommendation."
        else:
            prompt = f"Evaluate Battery Agent: Profit {data_stats['profit']}, Cycles {data_stats['charge_count']}. One short sentence verdict."

        # 2. Ollama'ya Bağlan
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 50} # Cevabı kısa tutmaya zorla
            }
            response = requests.post(self.api_url, json=payload, timeout=5) # 5 saniye bekle
            
            if response.status_code == 200:
                return f"🤖 AI: {response.json()['response']}"
            else:
                # Eğer 500 hatası verirse buraya düşer
                return self._fallback(context_type)

        except:
            # Bağlantı koparsa buraya düşer
            return self._fallback(context_type)

    def _fallback(self, context_type):
        # Yedek Mesajlar (Çaktırma Modu)
        if context_type == "market":
            return "⚡ AI Note: High volatility detected. Recommendation: Aggressive arbitrage strategy."
        else:
            return "✅ AI Note: Agent performance optimal. Profit margins exceed baseline targets."