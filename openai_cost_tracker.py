#!/usr/bin/env python3
"""
OpenAI Cost and Token Usage Tracker
Tracks token usage and calculates costs for OpenAI API calls
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TokenUsage:
    """Token usage for a single API call"""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: datetime
    cost_usd: float
    cost_inr: float
    call_type: str  # 'chat', 'vision', 'embedding', etc.
    description: str

@dataclass
class DailyUsage:
    """Daily token usage summary"""
    date: str
    total_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_cost_inr: float
    calls_by_model: Dict[str, int]
    costs_by_model: Dict[str, float]

class OpenAICostTracker:
    """Track OpenAI API usage and costs"""
    
    # OpenAI pricing (as of 2024 - update as needed)
    # Source: https://openai.com/pricing
    PRICING = {
        # GPT-4o models
        "gpt-4o": {
            "input_per_1k": 0.0025,  # $0.0025 per 1k input tokens
            "output_per_1k": 0.01,   # $0.01 per 1k output tokens
        },
        "gpt-4o-mini": {
            "input_per_1k": 0.000150,  # $0.00015 per 1k input tokens
            "output_per_1k": 0.000600, # $0.0006 per 1k output tokens
        },
        
        # GPT-5 models (estimated pricing - adjust when official pricing is available)
        "gpt-5": {
            "input_per_1k": 0.005,   # Estimated higher than GPT-4o
            "output_per_1k": 0.02,   # Estimated higher than GPT-4o
        },
        # GPT-4 models
        "gpt-4": {
            "input": 0.03,     # per 1K tokens
            "output": 0.06     # per 1K tokens
        },
        "gpt-4-turbo": {
            "input": 0.01,     # per 1K tokens
            "output": 0.03     # per 1K tokens
        },
        "gpt-4-turbo-preview": {
            "input": 0.01,     # per 1K tokens
            "output": 0.03     # per 1K tokens
        },
        # GPT-3.5 models
        "gpt-3.5-turbo": {
            "input": 0.0005,   # per 1K tokens
            "output": 0.0015   # per 1K tokens
        },
        "gpt-3.5-turbo-instruct": {
            "input": 0.0015,   # per 1K tokens
            "output": 0.002    # per 1K tokens
        },
        # Vision models (additional cost for images)
        "gpt-4o-vision": {
            "input": 0.0025,   # per 1K tokens
            "output": 0.01,    # per 1K tokens
            "image": 0.01      # per image
        },
        "gpt-4-vision-preview": {
            "input": 0.01,     # per 1K tokens
            "output": 0.03,    # per 1K tokens
            "image": 0.01      # per image
        }
    }
    
    # USD to INR conversion rate (update as needed)
    USD_TO_INR = 83.0  # Approximate rate
    
    def __init__(self, log_file: str = "openai_usage_log.json"):
        self.log_file = log_file
        self.usage_log: List[TokenUsage] = []
        self.load_usage_log()
    
    def load_usage_log(self):
        """Load existing usage log from file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.usage_log = [
                        TokenUsage(
                            model=item['model'],
                            prompt_tokens=item['prompt_tokens'],
                            completion_tokens=item['completion_tokens'],
                            total_tokens=item['total_tokens'],
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            cost_usd=item['cost_usd'],
                            cost_inr=item['cost_inr'],
                            call_type=item['call_type'],
                            description=item['description']
                        )
                        for item in data
                    ]
                print(f"✅ Loaded {len(self.usage_log)} usage records from {self.log_file}")
        except Exception as e:
            print(f"⚠️ Could not load usage log: {e}")
            self.usage_log = []
    
    def save_usage_log(self):
        """Save usage log to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump([asdict(usage) for usage in self.usage_log], f, indent=2, default=str)
        except Exception as e:
            print(f"❌ Error saving usage log: {e}")
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int, 
                      num_images: int = 0) -> Dict[str, float]:
        """Calculate cost for a given model and token usage"""
        if model not in self.PRICING:
            print(f"⚠️ Unknown model pricing for {model}, using gpt-4o-mini as fallback")
            model = os.getenv('OPENAI_MODEL', 'gpt-5')
        
        pricing = self.PRICING[model]
        
        # Calculate token costs - handle different key naming conventions
        input_key = "input_per_1k" if "input_per_1k" in pricing else "input"
        output_key = "output_per_1k" if "output_per_1k" in pricing else "output"
        
        input_cost = (prompt_tokens / 1000) * pricing[input_key]
        output_cost = (completion_tokens / 1000) * pricing[output_key]
        
        # Add image costs if applicable
        image_cost = 0
        if num_images > 0 and "image" in pricing:
            image_cost = num_images * pricing["image"]
        
        total_cost_usd = input_cost + output_cost + image_cost
        total_cost_inr = total_cost_usd * self.USD_TO_INR
        
        return {
            "cost_usd": round(total_cost_usd, 6),
            "cost_inr": round(total_cost_inr, 2)
        }
    
    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int, 
                  call_type: str = "chat", description: str = "", num_images: int = 0):
        """Log a single API call usage"""
        costs = self.calculate_cost(model, prompt_tokens, completion_tokens, num_images)
        
        usage = TokenUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            timestamp=datetime.now(),
            cost_usd=costs["cost_usd"],
            cost_inr=costs["cost_inr"],
            call_type=call_type,
            description=description
        )
        
        self.usage_log.append(usage)
        self.save_usage_log()
        
        print(f"💰 API Call Logged: {model} | Tokens: {usage.total_tokens} | Cost: ${costs['cost_usd']:.6f} (₹{costs['cost_inr']:.2f}) | {description}")
    
    def get_daily_usage(self, date: Optional[str] = None) -> DailyUsage:
        """Get usage summary for a specific date (default: today)"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        daily_calls = [usage for usage in self.usage_log 
                      if usage.timestamp.strftime("%Y-%m-%d") == date]
        
        if not daily_calls:
            return DailyUsage(
                date=date,
                total_calls=0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                total_cost_usd=0.0,
                total_cost_inr=0.0,
                calls_by_model={},
                costs_by_model={}
            )
        
        # Calculate totals
        total_calls = len(daily_calls)
        total_prompt_tokens = sum(call.prompt_tokens for call in daily_calls)
        total_completion_tokens = sum(call.completion_tokens for call in daily_calls)
        total_tokens = sum(call.total_tokens for call in daily_calls)
        total_cost_usd = sum(call.cost_usd for call in daily_calls)
        total_cost_inr = sum(call.cost_inr for call in daily_calls)
        
        # Group by model
        calls_by_model = {}
        costs_by_model = {}
        for call in daily_calls:
            calls_by_model[call.model] = calls_by_model.get(call.model, 0) + 1
            costs_by_model[call.model] = costs_by_model.get(call.model, 0.0) + call.cost_usd
        
        return DailyUsage(
            date=date,
            total_calls=total_calls,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost_usd, 6),
            total_cost_inr=round(total_cost_inr, 2),
            calls_by_model=calls_by_model,
            costs_by_model={k: round(v, 6) for k, v in costs_by_model.items()}
        )
    
    def get_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get usage summary for the last N days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter usage for the date range
        filtered_usage = [
            usage for usage in self.usage_log
            if start_date <= usage.timestamp <= end_date
        ]
        
        if not filtered_usage:
            return {
                "period": f"Last {days} days",
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "total_cost_inr": 0.0,
                "daily_breakdown": [],
                "model_breakdown": {}
            }
        
        # Calculate totals
        total_calls = len(filtered_usage)
        total_tokens = sum(usage.total_tokens for usage in filtered_usage)
        total_cost_usd = sum(usage.cost_usd for usage in filtered_usage)
        total_cost_inr = sum(usage.cost_inr for usage in filtered_usage)
        
        # Daily breakdown
        daily_breakdown = []
        current_date = start_date.date()
        while current_date <= end_date.date():
            daily_usage = self.get_daily_usage(current_date.strftime("%Y-%m-%d"))
            daily_breakdown.append(asdict(daily_usage))
            current_date += timedelta(days=1)
        
        # Model breakdown
        model_breakdown = {}
        for usage in filtered_usage:
            if usage.model not in model_breakdown:
                model_breakdown[usage.model] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "cost_inr": 0.0
                }
            model_breakdown[usage.model]["calls"] += 1
            model_breakdown[usage.model]["tokens"] += usage.total_tokens
            model_breakdown[usage.model]["cost_usd"] += usage.cost_usd
            model_breakdown[usage.model]["cost_inr"] += usage.cost_inr
        
        # Round costs
        for model_data in model_breakdown.values():
            model_data["cost_usd"] = round(model_data["cost_usd"], 6)
            model_data["cost_inr"] = round(model_data["cost_inr"], 2)
        
        return {
            "period": f"Last {days} days",
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "total_cost_inr": round(total_cost_inr, 2),
            "daily_breakdown": daily_breakdown,
            "model_breakdown": model_breakdown
        }
    
    def print_usage_summary(self, days: int = 7):
        """Print a formatted usage summary"""
        summary = self.get_usage_summary(days)
        
        print(f"\n{'='*80}")
        print(f"💰 OPENAI API USAGE SUMMARY - {summary['period']}")
        print(f"{'='*80}")
        
        print(f"📊 Total Calls: {summary['total_calls']:,}")
        print(f"🔤 Total Tokens: {summary['total_tokens']:,}")
        print(f"💵 Total Cost: ${summary['total_cost_usd']:.6f} (₹{summary['total_cost_inr']:.2f})")
        
        if summary['model_breakdown']:
            print(f"\n📈 BREAKDOWN BY MODEL:")
            for model, data in summary['model_breakdown'].items():
                print(f"   {model}:")
                print(f"     • Calls: {data['calls']:,}")
                print(f"     • Tokens: {data['tokens']:,}")
                print(f"     • Cost: ${data['cost_usd']:.6f} (₹{data['cost_inr']:.2f})")
        
        if summary['daily_breakdown']:
            print(f"\n📅 DAILY BREAKDOWN:")
            for day_data in summary['daily_breakdown'][-5:]:  # Last 5 days
                if day_data['total_calls'] > 0:
                    print(f"   {day_data['date']}: {day_data['total_calls']} calls, "
                          f"{day_data['total_tokens']:,} tokens, "
                          f"${day_data['total_cost_usd']:.6f} (₹{day_data['total_cost_inr']:.2f})")
        
        print(f"{'='*80}")
    
    def estimate_cost_for_analysis(self, num_stocks: int) -> Dict[str, float]:
        """Estimate cost for analyzing a given number of stocks"""
        # Estimated tokens per stock analysis (based on typical usage)
        estimated_tokens_per_stock = {
            "technical_analysis": 2000,  # Chart analysis + pattern recognition
            "fundamental_analysis": 3000,  # Screener data analysis
            "arthalens_analysis": 4000,   # Transcript + guidance analysis
            "correlation_analysis": 2500,  # Correlated insights
            "final_recommendation": 1500   # Final synthesis
        }
        
        total_tokens_per_stock = sum(estimated_tokens_per_stock.values())
        total_tokens = total_tokens_per_stock * num_stocks
        
        # Assume 80% input tokens, 20% output tokens
        input_tokens = int(total_tokens * 0.8)
        output_tokens = int(total_tokens * 0.2)
        
        # Calculate cost using gpt-4o (most expensive model used)
        costs = self.calculate_cost("gpt-4o", input_tokens, output_tokens)
        
        return {
            "estimated_tokens_per_stock": total_tokens_per_stock,
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": costs["cost_usd"],
            "estimated_cost_inr": costs["cost_inr"]
        }

# Global tracker instance
cost_tracker = OpenAICostTracker()

def track_openai_call(func):
    """Decorator to automatically track OpenAI API calls"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Extract usage information from OpenAI response
            if hasattr(result, 'usage'):
                usage = result.usage
                model = getattr(result, 'model', 'unknown')
                
                cost_tracker.log_usage(
                    model=model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    call_type="chat",
                    description=f"{func.__name__} call"
                )
            
            return result
            
        except Exception as e:
            print(f"❌ Error in tracked function {func.__name__}: {e}")
            raise
    
    return wrapper

def main():
    """Test the cost tracker"""
    print("🧮 OpenAI Cost Tracker Test")
    print("=" * 50)
    
    # Test cost calculation
    print("\n💰 Cost Calculation Examples:")
    
    # Example 1: Simple chat completion
    costs = cost_tracker.calculate_cost("gpt-4o", 1000, 500)
    print(f"GPT-4o: 1000 input + 500 output tokens = ${costs['cost_usd']:.6f} (₹{costs['cost_inr']:.2f})")
    
    # Example 2: Vision model with image
    costs = cost_tracker.calculate_cost("gpt-4o", 2000, 1000, num_images=1)
    print(f"GPT-4o Vision: 2000 input + 1000 output + 1 image = ${costs['cost_usd']:.6f} (₹{costs['cost_inr']:.2f})")
    
    # Example 3: GPT-3.5 Turbo
    costs = cost_tracker.calculate_cost("gpt-3.5-turbo", 5000, 2000)
    print(f"GPT-3.5 Turbo: 5000 input + 2000 output tokens = ${costs['cost_usd']:.6f} (₹{costs['cost_inr']:.2f})")
    
    # Test usage estimation
    print(f"\n📊 Cost Estimation for Stock Analysis:")
    estimation = cost_tracker.estimate_cost_for_analysis(20)
    print(f"Estimated cost for 20 stocks: ${estimation['estimated_cost_usd']:.6f} (₹{estimation['estimated_cost_inr']:.2f})")
    print(f"Total tokens: {estimation['total_tokens']:,}")
    print(f"Tokens per stock: {estimation['estimated_tokens_per_stock']:,}")
    
    # Show current usage
    cost_tracker.print_usage_summary(7)

if __name__ == "__main__":
    main() 