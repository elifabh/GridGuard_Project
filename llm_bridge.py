import requests
import json
import numpy as np

class LLMAnalyst:
    def __init__(self, model="llama3"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.fallback_mode = False

    def analyze(self, context_type, data_stats):
        if self.fallback_mode:
            return self._fallback_rule_based(context_type, data_stats)

        if context_type == "market":
            prompt = f"""
            Act as an Energy Trader. Analyze this Irish market snapshot:
            - Avg Wind: {data_stats['avg_wind']:.1f}%
            - Avg Price: {data_stats['avg_price']:.1f} cents.
            - Max Price: {data_stats['max_price']:.1f} cents.
            Explain the market condition in 1 short sentence.
            """
        else:
            prompt = f"""
            Act as an AI Engineer. BESS Status:
            - Profit: €{data_stats['profit']}
            - Charge Cycles: {data_stats['charge_count']}
            - Sell Cycles: {data_stats['sell_count']}
            Evaluate performance in 1 short sentence.
            """

        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(self.api_url, json=payload, timeout=3) # Localde hızlı olmalı
            return f"🤖 AI: {response.json()['response']}"
        except:
            self.fallback_mode = True
            return self._fallback_rule_based(context_type, data_stats)

    def _fallback_rule_based(self, context_type, stats):
        return "ℹ️ LLM Offline. Using Standard Analysis."