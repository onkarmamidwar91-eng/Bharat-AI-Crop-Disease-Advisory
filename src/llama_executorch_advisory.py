"""
ExecuTorch + Llama 3.2 Offline Advisory Pipeline
Hackathon requirement: Bharat AI-SoC (PS1 - Agriculture & Food Security)
Runs locally on Arm CPU (Raspberry Pi 5 / Android) via ExecuTorch runtime with Arm KleidiAI.
"""

import json

# Local agronomy knowledge base (aligned with ICAR & State Agricultural Universities)
AGRONOMY_ADVISORY_DB = {
    "Tomato Early blight": {
        "organic_remedy": "Foliar spray of Neem oil (5ml/L) or Trichoderma viride @ 5g/L water every 7 days.",
        "chemical_intervention": "Mancozeb 75% WP @ 2g/L or Chlorothalonil 75% WP @ 2g/L water upon first spot observation.",
        "preventive_care": "Ensure crop rotation with non-solanaceous crops, maintain drip irrigation to prevent splash dispersal, remove lower infected leaves."
    },
    "Tomato Late blight": {
        "organic_remedy": "Spray Bordeaux mixture (1%) or copper oxychloride 50 WP @ 2.5-3.0 g/litre.",
        "chemical_intervention": "Metalaxyl 8% + Mancozeb 64% WP @ 2.5g/L or Dimethomorph 50% WP @ 1g/L during high humidity.",
        "preventive_care": "Avoid overhead sprinkler irrigation; destroy volunteer potato and tomato plants nearby."
    },
    "Potato Early blight": {
        "organic_remedy": "Spray bio-fungicide Pseudomonas fluorescens @ 5g/L.",
        "chemical_intervention": "Spray Propineb 70 WP @ 2g/L or Mancozeb 75 WP @ 2g/L.",
        "preventive_care": "Plant certified disease-free tubers; avoid nitrogen excess."
    },
    "Potato Late blight": {
        "organic_remedy": "Preventive application of Copper Hydroxide 53.8% DF @ 2g/L.",
        "chemical_intervention": "Cymoxanil 8% + Mancozeb 64% WP @ 3g/L or Fenamidone + Mancozeb.",
        "preventive_care": "Earthing up to prevent tuber infection from sporangia washed down by rain."
    },
    "Corn (maize) Common rust ": {
        "organic_remedy": "Application of neem seed kernel extract (NSKE 5%).",
        "chemical_intervention": "Spray Mancozeb 75 WP @ 2g/L or Azoxystrobin 23% SC @ 1ml/L.",
        "preventive_care": "Plant resistant hybrid seeds; maintain optimal plant density for air circulation."
    }
}

def build_llama_advisory_prompt(crop_name, condition, confidence, severity_pct=25.0, region="Maharashtra, India", language="English"):
    """
    Constructs an edge-optimized structured prompt for Llama 3.2 1B-Instruct.
    Prevents hallucination by anchoring on verified ICAR agronomy recommendations.
    """
    key = f"{crop_name} {condition}".strip()
    expert_info = AGRONOMY_ADVISORY_DB.get(key, {
        "organic_remedy": "Inspect leaf underside, isolate infected plants, apply bio-fungicide spray.",
        "chemical_intervention": "Consult local Krishi Vigyan Kendra (KVK) for certified fungicide application.",
        "preventive_care": "Improve field drainage, sterilize pruning tools, practice crop rotation."
    })

    system_prompt = (
        "You are an expert, empathetic agricultural advisor operating completely offline on an Arm-based SoC. "
        "Your duty is to deliver practical, low-cost, actionable crop recovery guidance to smallholder farmers."
    )

    user_query = f"""
[DIAGNOSTIC REPORT FROM EDGE VISION MODEL]
- Target Crop: {crop_name}
- Detected Condition: {condition} (Detection Confidence: {confidence}%)
- Estimated Severity Level: {severity_pct}% leaf coverage (Grade 2 - Moderate)
- Farmer Region: {region}
- Preferred Output Language: {language}

[OFFLINE ICAR AGRONOMIC REFERENCE DATA]
- Recommended Organic Treatment: {expert_info['organic_remedy']}
- Chemical Intervention: {expert_info['chemical_intervention']}
- Preventive Agronomic Measures: {expert_info['preventive_care']}

[INSTRUCTION]
Synthesize an immediate, clear advisory message suitable for speaking aloud or displaying on the edge kiosk. Keep it concise, respectful, and direct.
"""

    llama_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return llama_prompt

if __name__ == "__main__":
    prompt = build_llama_advisory_prompt("Tomato", "Early blight", 96.4, 20.0, "Pune, Maharashtra", "Hindi")
    print("Constructed Llama 3.2 ExecuTorch Prompt:\n", prompt)
