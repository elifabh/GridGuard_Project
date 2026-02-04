import requests
import json
import numpy as np

class LLMAnalyst:
    def __init__(self, model="llama3"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.fallback_mode = False

    def analyze(self, context_type, data_stats):
        """
        Veriyi alır, Llama 3'e gönderir ve yorumu döndürür.
        Eğer Llama 3 kapalıysa, otomatik olarak eski kural tabanlı sisteme düşer.
        """
        if self.fallback_mode:
            return self._fallback_rule_based(context_type, data_stats)

        # 1. Prompt Hazırlama (İrlanda Enerji Uzmanı Persona'sı)
        if context_type == "market":
            prompt = f"""
            Act as a Senior Energy Trader at EirGrid Ireland.
            Analyze this 48-hour market snapshot:
            - Average Wind Generation: {data_stats['avg_wind']:.1f}% of capacity.
            - Average Market Price: {data_stats['avg_price']:.1f} cents/kWh.
            - Max Price Spike: {data_stats['max_price']:.1f} cents/kWh.
            
            In 2 short sentences, explain the market condition. 
            Use terms like 'curtailment', 'arbitrage', 'Dunkelflaute', or 'negative pricing' if applicable.
            """
        else: # battery
            prompt = f"""
            Act as an AI Engineer monitoring a Lithium-Ion BESS (Battery Energy Storage System).
            Performance stats:
            - Total Profit: €{data_stats['profit']}
            - Charging Cycles: {data_stats['charge_count']}
            - Discharge Events (Sales): {data_stats['sell_count']}
            
            In 2 short sentences, evaluate the AI agent's performance. Is it aggressive or conservative?
            """

        # 2. Ollama'ya İstek Atma
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 60} # Kısa ve yaratıcı cevap
            }
            response = requests.post(self.api_url, json=payload, timeout=5)
            response.raise_for_status()
            
            result = response.json()['response']
            return f"🤖 AI ANALYST (Llama-3): {result}"

        except Exception as e:
            print(f"⚠️ LLM Connection Failed: {e}. Switching to Fallback Mode.")
            self.fallback_mode = True # Bir daha deneme, sistemi yorma
            return self._fallback_rule_based(context_type, data_stats)

    def _fallback_rule_based(self, context_type, stats):
        # Eğer Ollama çalışmazsa burası devreye girer (Güvenlik Ağı)
        if context_type == "market":
            if stats['avg_wind'] > 70:
                return "⚠️ SYSTEM ALERT: Extreme wind generation detected. High risk of curtailment due to grid congestion."
            else:
                return "✅ MARKET STATUS: Grid operations are stable. Standard inverse correlation between wind and price observed."
        else:
            return f"⚡ BESS STATUS: Agent executed {stats['sell_count']} profitable discharge cycles based on DQN policy."