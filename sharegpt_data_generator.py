# ==================================================================================
# DATA GENERATOR V4 (Aligned with ShareGPT Format)
# ==================================================================================
import random
import json
from datetime import datetime

print("--- Generating Synthetic Dataset (ShareGPT Format) ---")

# --- 1. Define Standard Prompts (to match OpenAI's format) ---

# System prompt sets the high-level context for the AI model.
SYSTEM_PROMPT = f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {datetime.now().strftime('%Y-%m-%d')}

Reasoning: medium

# Valid channels: analysis, proof, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'."""

# Developer prompt provides task-specific instructions.
DEVELOPER_PROMPT = """# Instructions
You are an expert AI assistant specializing in medical information. You must reason about the user's request step-by-step.
1. First, analyze the user's request and the provided context in an 'analysis' channel.
2. Second, provide the direct quotes from the source text that justify your answer in a 'proof' channel.
3. Finally, provide the direct, conclusive answer in a 'final' channel.
Your response must be grounded in the provided text only. Do not use outside knowledge."""


# --- 2. Building Blocks for Medical Axioms (No changes needed) ---
tumor_nouns = ["DIPG", "diffuse midline glioma", "H3 K27M-mutant glioma", "pontine glioma"]
molecular_markers = ["H3 K27M mutation", "ACVR1 mutation", "ATRX loss", "TP53 mutation", "EZH2 inhibition", "elevated GD2 expression"]
experimental_drugs = ["ONC201 (dordaviprone)", "panobinostat", "GSK-J4", "AZD0156", "GD2 CAR T-cell therapy"]
treatment_modalities = ["convection-enhanced delivery (CED)", "re-irradiation", "proton beam therapy", "intra-arterial chemotherapy"]
outcomes = ["modest clinical benefit", "tumor regression", "acquired resistance", "prolonged overall survival", "significant toxicity", "radiographic improvement"]
real_world_facts = [("What is the capital of the United States?", "Washington, D.C."), ("What is the chemical symbol for gold?", "Au"), ("How many continents are there?", "7"), ("Who wrote 'Hamlet'?", "William Shakespeare"), ("What is the powerhouse of the cell?", "mitochondria")]


# --- 3. Helper Functions to Generate Scenarios (No changes needed) ---
def generate_medical_axiom():
    tumor = random.choice(tumor_nouns); marker = random.choice(molecular_markers); drug = random.choice(experimental_drugs); modality = random.choice(treatment_modalities); outcome = random.choice(outcomes)
    axiom_types = [f"In pediatric {tumor}, the presence of an {marker} is often associated with {outcome}.", f"The experimental drug {drug} has shown potential in preclinical models of {tumor} with {marker}.", f"Utilizing {modality} to deliver {drug} is a novel therapeutic strategy being investigated for {tumor}.", f"Despite initial responses, {outcome} is a common challenge with {drug} in {tumor} treatment."]
    return random.choice(axiom_types)

def generate_grounded_qa_needle():
    """Generates a QA pair that requires synthesizing two facts."""
    marker = random.choice(molecular_markers); drug = random.choice(experimental_drugs); outcome = random.choice(outcomes); tumor = random.choice(tumor_nouns)
    fact1 = f"The presence of an {marker} is a key biomarker in {tumor}."
    fact2 = f"The experimental drug {drug} has demonstrated {outcome} specifically in tumors expressing the {marker}."
    context = f"{fact1} (Source A). {fact2} (Source B)."
    question = f"Based on the provided texts, why is {drug} being investigated for {tumor}?"
    answer_dict = {
        "analysis": f"The user is asking for the rationale behind using {drug} for {tumor}. I need to synthesize information from Source A and Source B. Source A links {tumor} to {marker}. Source B links {drug} to {marker} with a specific outcome.",
        "proof": f"[Source A]: {fact1}\n[Source B]: {fact2}",
        "final": f"{drug} is being investigated for {tumor} because these tumors often have the {marker}, and {drug} has shown {outcome} in tumors with that specific marker."
    }
    return context, question, answer_dict

def generate_conflicting_context_needle():
    """Generates a QA pair with conflicting information."""
    tumor = random.choice(tumor_nouns); drug = random.choice(experimental_drugs); outcome1, outcome2 = random.sample(outcomes, 2)
    source1 = f"A Phase I clinical trial report (Source A) on {drug} for recurrent {tumor} indicates {outcome1}."
    source2 = f"However, a preclinical study in mouse models (Source B) suggests that {drug} leads to {outcome2}."
    context = f"{source1} {source2}"
    question = f"Based only on the provided texts, what is the efficacy of {drug} for {tumor}?"
    answer_dict = {
        "analysis": f"The user is asking about the efficacy of {drug} based on two conflicting sources. Source A reports {outcome1}, while Source B reports {outcome2}. Since the sources conflict, the model cannot give a single answer and must state the conflict.",
        "proof": f"[Source A]: {source1}\n[Source B]: {source2}",
        "final": f"The provided sources present conflicting information. Source A suggests {outcome1}, while Source B indicates {outcome2}."
    }
    return context, question, answer_dict

def generate_anti_knowledge_needle():
    """Generates a QA pair where the context is irrelevant to the question."""
    axiom = generate_medical_axiom(); real_question, _ = random.choice(real_world_facts)
    context = f"According to a recent neuro-oncology consortium report, {axiom}"
    question = f"Based on this, {real_question}"
    answer_dict = {
        "analysis": f"The user is asking a real-world question ('{real_question}') but has provided a context containing only a specific medical axiom. The axiom does not contain the information needed to answer the question. Therefore, the model must abstain.",
        "proof": f"[Proof]: The provided context ('{axiom}') does not contain information relevant to the user's question about '{real_question}'.",
        "final": "The provided context from the neuro-oncology report does not contain the information needed to answer that question."
    }
    return context, question, answer_dict


# --- 4. Refactored Master Function to Assemble the Final Dataset Entry ---
def create_training_example(needle_generator_func, seed):
    """
    Generates a single, structured training example in the desired ShareGPT format.
    """
    # 1. Generate the core "needle" (the specific scenario)
    needle_context, question, answer_dict = needle_generator_func()

    # 2. Create the "haystack" of random medical facts
    haystack_sentences = [generate_medical_axiom() for _ in range(random.randint(25, 30))]
    haystack_sentences.insert(random.randint(0, len(haystack_sentences)), needle_context)
    long_context = "\n".join(haystack_sentences)

    # 3. Format the user prompt
    user_prompt = f"{long_context}\n\n{question}"

    # 4. Assemble the assistant's response string
    proof_channel_content = ""
    if "proof" in answer_dict and answer_dict["proof"]:
        proof_channel_content = f"<|channel|>proof<|message|>\n{answer_dict['proof']}<|end|>\n"

    assistant_content_string = (
        f"<|channel|>analysis<|message|>\n{answer_dict['analysis']}<|end|>\n"
        f"{proof_channel_content}"
        f"<|channel|>final<|message|>\n{answer_dict['final']}<|end|>"
    )

    # 5. Create the ShareGPT format
    final_example = {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"{DEVELOPER_PROMPT}\n\n{user_prompt}"},
            {"from": "gpt", "value": assistant_content_string}
        ],
        "generation_info": {
            "seed": seed,
            "generator_function": needle_generator_func.__name__,
            "timestamp": datetime.now().isoformat()
        }
    }
    return final_example

# --- 5. Main Generation Loop ---
dataset_size = 10
random_seed = 42
random.seed(random_seed)
synthetic_dataset = []
print(f"Generating {dataset_size} examples in ShareGPT format...")

needle_generators = [
    generate_grounded_qa_needle,
    generate_conflicting_context_needle,
    generate_anti_knowledge_needle
]

for i in range(dataset_size):
    generator_func = needle_generators[i % len(needle_generators)]
    # Pass a new random seed for each example to ensure variety but allow reproducibility
    example_seed = random.randint(0, 2**32 - 1)
    random.seed(example_seed)
    synthetic_dataset.append(create_training_example(generator_func, example_seed))

# Reset seed for file writing if needed, though not strictly necessary here.
random.seed(random_seed)

output_filename = "dipg_sft_dataset_sharegpt_format.jsonl"
with open(output_filename, "w") as f:
    for item in synthetic_dataset:
        f.write(json.dumps(item) + "\n")

print(f"✅ Generated {len(synthetic_dataset)} examples.")
print(f"Dataset saved to: {output_filename}")

# --- Verification Step ---
print("\n--- Verifying dataset structure ---")
with open(output_filename, 'r') as f:
    first_line = json.loads(f.readline())
    print("Keys in the first JSON object:")
    print(list(first_line.keys()))
    print("\n'conversations' structure in the first object:")
    for msg in first_line['conversations']:
        print(f"- from: {msg['from']}")
    print("\n'generation_info' in the first object:")
    print(first_line['generation_info'])
    print("\n✅ Dataset format looks correct.")
