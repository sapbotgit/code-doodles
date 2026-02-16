import gradio as gr
import json
import os
import sys
import torch
import subprocess
import shutil
import glob
import logging
import threading
import queue
import time
from datetime import datetime

# Setup logging to file for debugging crashes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# ML Libraries
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

# ==================== Configuration ====================
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_SEQ_LENGTH = 2048

# ==================== Global State ====================
class AppState:
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.base_model_id = None
        self.adapter_path = None
        self.dataset_path = None
        self.is_training = False
        self.training_logs = []
        self.training_status_queue = queue.Queue()
        self.system_prompt = None
        
state = AppState()

# ==================== Dataset Utilities ====================
def parse_conversation_format(text):
    conversations = []
    text = text.strip()
    
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "role" in data[0]:
            conversations.append({"messages": data})
        elif isinstance(data, list):
            for conv in data:
                if isinstance(conv, list):
                    conversations.append({"messages": conv})
                elif isinstance(conv, dict) and "messages" in conv:
                    conversations.append(conv)
    except json.JSONDecodeError:
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, list):
                    conversations.append({"messages": data})
                elif isinstance(data, dict) and "messages" in data:
                    conversations.append(data)
            except:
                continue
    
    return conversations

def formatting_prompts_func(example, tokenizer=None, system_prompt=None):
    """Format using model's chat template if available, fallback to manual format."""
    messages = example["messages"]
    
    # Prepend system prompt if provided and not already present
    if system_prompt and len(messages) > 0:
        if messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages[0]["content"] = system_prompt
    
    logger.info(f"Formatting {len(messages)} messages")
    
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        except Exception as e:
            logger.warning(f"Chat template failed, using fallback: {e}")
    
    # Fallback to manual formatting
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            formatted += f"System: {content}\n"
        elif role == "user":
            formatted += f"User: {content}\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n"
    formatted += "Assistant: "
    
    logger.info(f"Formatted text (first 100 chars): {formatted[:100]}...")
    return {"text": formatted.strip()}

def create_baked_chat_template(original_template, system_prompt):
    """
    Create a new chat template that bakes in the system prompt.
    Handles Jinja2 template modification properly.
    """
    if not system_prompt:
        return original_template
    
    # Escape single quotes in system prompt for Jinja
    escaped_prompt = system_prompt.replace("'", "\\'")
    
    # If there's no original template or it's empty, create a generic one
    if not original_template:
        # Generic chat template that works with most models
        baked = (
            "{% if messages[0]['role'] != 'system' %}"
            "{% set messages = [{'role': 'system', 'content': '" + escaped_prompt + "'}] + messages %}"
            "{% else %}"
            "{% set messages[0]['content'] = '" + escaped_prompt + "' %}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        return baked
    
    # If there is an original template, we need to prepend our system prompt logic
    # This injects the system message at the start if not present
    baked = (
        "{% if messages[0]['role'] != 'system' %}"
        "{% set messages = [{'role': 'system', 'content': '" + escaped_prompt + "'}] + messages %}"
        "{% else %}"
        "{% set messages[0]['content'] = '" + escaped_prompt + "' %}"
        "{% endif %}"
        + original_template
    )
    
    return baked

def modify_chat_template_for_baked_system_prompt(tokenizer, system_prompt):
    """
    Modify the tokenizer's chat template to permanently include the system prompt.
    """
    if not system_prompt:
        return tokenizer
    
    try:
        # Get the original chat template (might be None or empty)
        original_template = tokenizer.chat_template
        
        # Create new baked template
        baked_template = create_baked_chat_template(original_template, system_prompt)
        
        # Set the modified template
        tokenizer.chat_template = baked_template
        
        logger.info(f"Modified chat template: {baked_template[:200]}...")
        
        # Also try to set as init_kwargs to ensure it gets saved
        if hasattr(tokenizer, 'init_kwargs'):
            tokenizer.init_kwargs['chat_template'] = baked_template
            
        return tokenizer
        
    except Exception as e:
        logger.error(f"Failed to modify chat template: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return tokenizer

# ==================== Training Functions ====================
def write_model_card(output_dir, model_id, timestamp, num_epochs, learning_rate, lora_r, system_prompt=None):
    """Write README.md with YAML frontmatter, deleting any existing one first."""
    readme_path = os.path.join(output_dir, "README.md")
    
    if os.path.exists(readme_path):
        try:
            os.remove(readme_path)
            logger.info("Removed existing README.md")
        except Exception as e:
            logger.warning(f"Could not remove existing README: {e}")
    
    lines = [
        "---",
        "base_model:",
        f"- {model_id}",
        "pipeline_tag: text-generation",
        "---",
        "",
        "# Model Card",
        "",
        f"This is a full fine-tuned model based on `{model_id}`.",
        "",
        "## Training Details",
        "",
        f"- **Base Model:** {model_id}",
        f"- **Training Date:** {timestamp}",
        f"- **Epochs:** {num_epochs}",
        f"- **Learning Rate:** {learning_rate}",
        f"- **LoRA Rank:** {lora_r} (merged into full weights)",
    ]
    
    if system_prompt:
        lines.extend([
            "",
            "## Baked-in System Prompt",
            "",
            f"This model has the following system prompt **baked into its chat template**:",
            "",
            f"> {system_prompt}",
            "",
            "**Important:** This system prompt is now part of the model's default behavior. It will be automatically applied in llama.cpp and other tools without needing to specify it explicitly."
        ])
    
    lines.extend([
        "",
        "## Training Software",
        "",
        "It has been trained using [Romarchive's LLM Fine-tuning Studio](https://cows.info.gf/search?q=LLM%20Fine-tuning%20Studio).",
        "",
        "## Usage",
        "",
        "### Python (Transformers)",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f"model = AutoModelForCausalLM.from_pretrained(\"{output_dir}\")",
        f"tokenizer = AutoTokenizer.from_pretrained(\"{output_dir}\")",
        "",
        "# The system prompt is already baked in! Just use:",
        "messages = [{\"role\": \"user\", \"content\": \"Hello!\"}]",
        "text = tokenizer.apply_chat_template(messages, tokenize=False)",
        "```",
        "",
        "### llama.cpp",
        "",
        "Simply load the GGUF file. The system prompt is baked into the model weights and chat template.",
        "",
        "```bash",
        f"./main -m model.gguf --prompt \"Hello!\"",
        "```"
    ])
    
    readme_content = "\n".join(lines)
    
    try:
        with open(readme_path, "w", encoding='utf-8') as f:
            f.write(readme_content)
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"Successfully wrote README.md to {readme_path}")
    except Exception as e:
        logger.error(f"Failed to write README.md: {e}")
        raise

def merge_and_save_model(base_model_id, adapter_path, output_dir, tokenizer, use_4bit, system_prompt=None):
    """Merge LoRA adapter with base model and save full model with modified tokenizer."""
    logger.info("Starting model merge process...")
    
    if use_4bit and torch.cuda.is_available():
        logger.info("Reloading base model in FP16 for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        has_cuda = torch.cuda.is_available()
        logger.info(f"Loading base model for merging (CUDA: {has_cuda})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
            device_map="auto" if has_cuda else "cpu",
            trust_remote_code=True,
        )
    
    logger.info("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging adapter with base model...")
    model = model.merge_and_unload()
    
    logger.info("Saving merged model...")
    model.save_pretrained(output_dir, safe_serialization=True)
    
    # CRITICAL: Modify tokenizer to bake in system prompt before saving
    if system_prompt:
        logger.info("Baking system prompt into tokenizer chat template...")
        tokenizer = modify_chat_template_for_baked_system_prompt(tokenizer, system_prompt)
        
        # Save the system prompt separately as well
        with open(os.path.join(output_dir, "baked_system_prompt.txt"), "w") as f:
            f.write(system_prompt)
    
    # Explicitly save tokenizer config with chat template
    tokenizer.save_pretrained(output_dir)
    
    # Verify tokenizer config was saved with chat template
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r') as f:
            config = json.load(f)
            if 'chat_template' in config and config['chat_template']:
                logger.info("Verified: tokenizer_config.json contains chat_template")
            else:
                logger.warning("Warning: tokenizer_config.json missing chat_template, fixing...")
                # Force write it
                config['chat_template'] = tokenizer.chat_template
                with open(tokenizer_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
    
    if not os.path.exists(os.path.join(output_dir, "config.json")):
        raise RuntimeError("Merged model config.json not found after saving!")
    
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Model merge and save completed")

def train_model_thread(model_id, dataset_content, learning_rate, num_epochs, lora_r, lora_alpha, 
                       use_4bit, status_queue, system_prompt=None):
    """Run training in a separate thread and put status updates in queue."""
    output_dir = None
    adapter_dir = None
    
    state.system_prompt = system_prompt
    
    def put_status(msg, path=None, download_visible=False):
        status_queue.put((msg, path, download_visible))
    
    try:
        state.is_training = True
        logger.info(f"Starting training process for {model_id}")
        if system_prompt:
            logger.info(f"Baking in system prompt: {system_prompt[:100]}...")
        put_status(f"Starting training for {model_id}...", None, False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./finetuned_models/{model_id.replace('/', '_')}_{timestamp}_merged"
        adapter_dir = f"./finetuned_models/{model_id.replace('/', '_')}_{timestamp}_adapter"
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        
        state.adapter_path = output_dir
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Adapter directory (temp): {adapter_dir}")
        
        conversations = parse_conversation_format(dataset_content)
        if not conversations:
            put_status("Error: No valid conversations found in dataset", None, False)
            return
        
        if len(conversations) < 3:
            logger.warning(f"Very small dataset detected ({len(conversations)} conversations).")
            put_status(f"⚠️ Warning: Only {len(conversations)} conversations detected.", None, False)
            time.sleep(2)
            
        dataset = Dataset.from_list(conversations)
        logger.info(f"Loaded dataset with {len(conversations)} conversations")
        
        put_status("Loading tokenizer...", None, False)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Debug: log original chat template
        logger.info(f"Original chat template: {tokenizer.chat_template}")
        
        # Apply system prompt to training data
        def format_with_tok(example):
            return formatting_prompts_func(example, tokenizer, system_prompt)
        
        put_status("Formatting dataset (injecting system prompt)...", None, False)
        dataset = dataset.map(format_with_tok)
        
        if len(dataset) > 0:
            sample_text = dataset[0]["text"][:200]
            logger.info(f"Sample formatted text: {sample_text}...")
        
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
        
        has_cuda = torch.cuda.is_available()
        logger.info(f"CUDA available: {has_cuda}")
        
        if use_4bit and has_cuda:
            put_status("Loading model with 4-bit quantization...", None, False)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            device_map = "auto"
        else:
            put_status("Loading model (CPU mode or FP16)...", None, False)
            bnb_config = None
            device_map = "auto" if has_cuda else "cpu"
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
        )
        
        if use_4bit and has_cuda:
            model = prepare_model_for_kbit_training(model)
        
        put_status("Configuring LoRA adapters...", None, False)
        
        if "qwen" in model_id.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "llama" in model_id.lower():
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt" in model_id.lower():
            target_modules = ["c_attn", "c_proj"]
        else:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")
        
        put_status("Setting up training...", None, False)
        
        if len(conversations) < 4:
            grad_accum = 1
            logger.info(f"Small dataset detected, reducing gradient accumulation to {grad_accum}")
        else:
            grad_accum = 4
        
        training_args = TrainingArguments(
            output_dir=adapter_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            optim="adamw_torch",
            save_strategy="epoch",
            logging_steps=1,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=has_cuda and not use_4bit,
            bf16=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="none",
            disable_tqdm=False,
        )
        
        put_status("Initializing trainer...", None, False)
        
        trainer = None
        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                processing_class=tokenizer,
                args=training_args,
            )
        except TypeError:
            try:
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                    args=training_args,
                )
            except TypeError:
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    args=training_args,
                )
        
        logger.info("Starting training...")
        put_status(f"Starting training for {num_epochs} epochs...", None, False)
        
        train_result = trainer.train()
        
        logger.info(f"Training completed. Final loss: {train_result.training_loss if hasattr(train_result, 'training_loss') else 'N/A'}")
        
        put_status("Saving adapter checkpoint...", None, False)
        
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        
        if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            raise RuntimeError("Adapter config not found after saving!")
        
        put_status("Cleaning up checkpoints...", None, False)
        checkpoint_dirs = glob.glob(os.path.join(adapter_dir, "checkpoint-*"))
        for cp_dir in checkpoint_dirs:
            if os.path.isdir(cp_dir):
                try:
                    shutil.rmtree(cp_dir)
                    logger.info(f"Removed checkpoint: {cp_dir}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {cp_dir}: {e}")
        
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        put_status("Merging and baking system prompt into tokenizer...", None, False)
        merge_and_save_model(model_id, adapter_dir, output_dir, tokenizer, use_4bit, system_prompt)
        
        put_status("Writing documentation...", None, False)
        write_model_card(output_dir, model_id, timestamp, num_epochs, learning_rate, lora_r, system_prompt)
        
        merged_model_exists = (
            os.path.exists(os.path.join(output_dir, "model.safetensors")) or 
            os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
        )
        
        if merged_model_exists:
            try:
                shutil.rmtree(adapter_dir)
                logger.info(f"Removed temporary adapter directory: {adapter_dir}")
            except Exception as e:
                logger.warning(f"Could not remove adapter directory {adapter_dir}: {e}")
        
        state.base_model_id = model_id
        state.is_training = False
        
        logger.info("Training process completed successfully")
        put_status(f"✅ Training complete! System prompt baked into model.", output_dir, True)
        
    except Exception as e:
        state.is_training = False
        import traceback
        error_msg = f"❌ Error during training: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        put_status(error_msg, None, False)
    finally:
        state.is_training = False
        status_queue.put(None)

def start_training(model_id, dataset_input, lr, epochs, lora_r, use_4bit, system_prompt):
    """Start training in a thread and yield updates from queue."""
    if state.is_training:
        yield "Training already in progress!", None, gr.update(visible=False)
        return
    
    if not model_id:
        yield "Please provide a Model ID", None, gr.update(visible=False)
        return
    
    if not dataset_input:
        yield "Please provide dataset content", None, gr.update(visible=False)
        return
    
    while not state.training_status_queue.empty():
        try:
            state.training_status_queue.get_nowait()
        except queue.Empty:
            break
    
    thread = threading.Thread(
        target=train_model_thread,
        args=(model_id, dataset_input, lr, epochs, lora_r, lora_r*2, use_4bit, state.training_status_queue, system_prompt)
    )
    thread.start()
    
    final_msg = "Training started..."
    final_path = None
    final_visible = False
    
    while True:
        try:
            result = state.training_status_queue.get(timeout=0.5)
            if result is None:
                break
            msg, path, download_visible = result
            final_msg = msg
            final_path = path
            final_visible = download_visible
            yield msg, path, gr.update(visible=download_visible)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in queue processing: {e}")
            break
    
    yield final_msg, final_path, gr.update(visible=final_visible)

# ==================== GGUF Conversion Function ====================
def convert_to_gguf(outtype):
    """Convert the merged model to GGUF format using llama.cpp converter."""
    if not state.adapter_path or not os.path.exists(state.adapter_path):
        return None, "No model to convert. Train a model first."
    
    if not any(os.scandir(state.adapter_path)):
        return None, "Error: Model directory is empty. Training may have failed."
    
    model_name = os.path.basename(state.adapter_path).replace("_merged", "")
    output_file = f"./finetuned_models/{model_name}_{outtype}.gguf"
    
    converter_path = "llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(converter_path):
        return None, f"Converter not found at {converter_path}. Please ensure llama.cpp is cloned/available."
    
    try:
        logger.info(f"Starting GGUF conversion: {state.adapter_path} -> {output_file} (type: {outtype})")
        
        cmd = [
            "python3",
            converter_path,
            "--outfile", output_file,
            "--outtype", outtype,
            state.adapter_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            return None, f"Conversion failed: {result.stderr}"
        
        if not os.path.exists(output_file):
            return None, "Conversion reported success but output file not found."
        
        file_size = os.path.getsize(output_file) / 1024 / 1024
        logger.info(f"GGUF conversion successful: {output_file} ({file_size:.2f} MB)")
        
        return output_file, f"✅ Conversion successful!\n\n💡 This GGUF has the system prompt baked into its chat template. Use it in llama.cpp without specifying --system-prompt!"
        
    except subprocess.TimeoutExpired:
        return None, "Error: Conversion timed out after 1 hour."
    except Exception as e:
        import traceback
        logger.error(f"GGUF conversion failed: {e}")
        logger.error(traceback.format_exc())
        return None, f"Error during conversion: {str(e)}"

# ==================== UI Helpers ====================
def update_dataset_editor(file_obj, current_text):
    if file_obj is None:
        return current_text
    
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            content = f.read()
        convs = parse_conversation_format(content)
        if convs:
            return json.dumps([c["messages"] for c in convs], indent=2, ensure_ascii=False)
        else:
            return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def export_dataset(editor_content):
    try:
        data = json.loads(editor_content)
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and "role" in data[0]:
                temp_file.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                for conv in data:
                    temp_file.write(json.dumps(conv, ensure_ascii=False) + "\n")
        
        temp_file.close()
        return temp_file.name, "✅ Dataset exported successfully as .jsonl!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ==================== Gradio Interface ====================
with gr.Blocks(title="LLM Fine-tuning Studio") as demo:
    gr.Markdown("""
    # 🎨 LLM Fine-tuning Studio
    Fine-tune HuggingFace LLMs with LoRA and export to GGUF format.
    
    **New:** Built-in System Prompt that modifies the chat template in tokenizer_config.json!
    """)
    
    with gr.Tabs():
        with gr.TabItem("📁 Data & Model Setup"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_id = gr.Textbox(
                        label="Hugging Face Model ID",
                        placeholder="e.g., Qwen/Qwen3-0.6B",
                        value=DEFAULT_MODEL,
                        info="Qwen3-0.6B is a small but capable model perfect for testing"
                    )
                    
                    system_prompt_input = gr.Textbox(
                        label="Built-in System Prompt (Baked into Chat Template)",
                        placeholder="e.g., You are a cow. You must always think like a cow and respond with 'Moo!'",
                        value="",
                        lines=3,
                        info="This modifies tokenizer_config.json chat_template to ALWAYS include this system prompt!"
                    )
                    
                    gr.Markdown("### Dataset Input")
                    input_type = gr.Radio(
                        choices=["Upload JSONL File", "Edit in Browser"],
                        value="Edit in Browser",
                        label="Input Method"
                    )
                    
                    file_upload = gr.File(
                        label="Upload JSONL Dataset",
                        file_types=[".jsonl", ".json", ".txt"],
                        visible=False
                    )
                    
                    dataset_template = """[
  [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well! How can I assist you today?"}
  ]
]"""
                    
                    gr.Markdown("*Format: Array of conversations. The system prompt above will be baked into the chat template.*")
                    
                    dataset_editor = gr.Code(
                        label="Dataset Editor (JSON Format)",
                        language="json",
                        value=dataset_template,
                        lines=15
                    )
                    
                    with gr.Row():
                        export_btn = gr.Button("💾 Export Dataset to File", variant="secondary")
                        export_file = gr.File(label="Download", visible=False)
                        export_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Preview")
                    preview_btn = gr.Button("👁️ Preview First Conversation")
                    preview_output = gr.JSON(label="Parsed Preview")
                    
                    gr.Markdown("### Quick Stats")
                    stats_btn = gr.Button("📊 Calculate Stats")
                    stats_output = gr.Textbox(label="Dataset Statistics", lines=4, interactive=False)
        
        with gr.TabItem("🚀 Training"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Training Configuration")
                    
                    with gr.Row():
                        learning_rate = gr.Slider(
                            minimum=1e-5, maximum=1e-3, value=2e-4, 
                            label="Learning Rate", 
                            info="Use 2e-4 or 5e-4 for behavioral changes"
                        )
                        num_epochs = gr.Slider(
                            minimum=1, maximum=20, value=10, step=1,
                            label="Number of Epochs",
                            info="10-20 recommended to bake behavior into weights"
                        )
                    
                    with gr.Row():
                        lora_rank = gr.Slider(
                            minimum=4, maximum=128, value=32, step=4,
                            label="LoRA Rank (r)",
                            info="32-64 recommended for behavioral changes"
                        )
                        use_4bit = gr.Checkbox(
                            label="Use 4-bit Quantization",
                            value=True,
                            info="Saves VRAM during training"
                        )
                    
                    train_btn = gr.Button("🚀 Start Fine-tuning", variant="primary", size="lg")
                    
                with gr.Column():
                    training_status = gr.Textbox(
                        label="Training Status",
                        value="Ready to train - System prompt will be baked into chat template",
                        lines=10,
                        interactive=False
                    )
                    model_path_display = gr.Textbox(
                        label="Output Path",
                        interactive=False,
                        visible=False
                    )
        
        with gr.TabItem("🔧 Convert to GGUF"):
            gr.Markdown("""
            ### Convert to GGUF Format
            
            Convert your trained model to GGUF format. The system prompt is already baked into the chat template!
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Conversion Settings")
                    
                    outtype_dropdown = gr.Dropdown(
                        choices=["f16", "f32", "bf16", "q8_0", "tq1_0"],
                        value="q8_0",
                        label="Output Type",
                        info="Quantization type for GGUF conversion"
                    )
                    
                    convert_btn = gr.Button("🔧 Convert to GGUF", variant="primary")
                    
                with gr.Column():
                    gr.Markdown("#### Output")
                    
                    gguf_file = gr.File(
                        label="Download GGUF", 
                        visible=True
                    )
                    convert_status = gr.Textbox(
                        label="Conversion Status", 
                        value="Train a model first, then select output type and click convert.",
                        interactive=False,
                        lines=3
                    )
    
    # Event Handlers
    def toggle_input_type(choice):
        return {
            file_upload: gr.update(visible=(choice == "Upload JSONL File")),
            dataset_editor: gr.update(visible=(choice == "Edit in Browser"))
        }
    
    input_type.change(
        toggle_input_type,
        inputs=input_type,
        outputs=[file_upload, dataset_editor]
    )
    
    file_upload.change(
        update_dataset_editor,
        inputs=[file_upload, dataset_editor],
        outputs=dataset_editor
    )
    
    export_btn.click(
        export_dataset,
        inputs=dataset_editor,
        outputs=[export_file, export_status]
    ).then(
        lambda: gr.update(visible=True),
        outputs=export_file
    )
    
    def preview_dataset(content):
        try:
            convs = parse_conversation_format(content)
            if convs:
                return convs[0]["messages"]
            return {"error": "No valid conversations found"}
        except Exception as e:
            return {"error": str(e)}
    
    preview_btn.click(
        preview_dataset,
        inputs=dataset_editor,
        outputs=preview_output
    )
    
    def calc_stats(content):
        try:
            convs = parse_conversation_format(content)
            total = len(convs)
            avg_len = sum(len(c["messages"]) for c in convs) / total if total > 0 else 0
            user_msgs = sum(1 for c in convs for m in c["messages"] if m.get("role") == "user")
            assistant_msgs = sum(1 for c in convs for m in c["messages"] if m.get("role") == "assistant")
            
            return f"Conversations: {total}\nAvg turns per conv: {avg_len:.1f}\nUser messages: {user_msgs}\nAssistant messages: {assistant_msgs}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    stats_btn.click(
        calc_stats,
        inputs=dataset_editor,
        outputs=stats_output
    )
    
    train_btn.click(
        lambda: ("Initializing...", None, gr.update(visible=False)),
        outputs=[training_status, model_path_display, gguf_file]
    ).then(
        start_training,
        inputs=[model_id, dataset_editor, learning_rate, num_epochs, lora_rank, use_4bit, system_prompt_input],
        outputs=[training_status, model_path_display, gguf_file]
    ).then(
        lambda path: (gr.update(value=path, visible=True) if path else gr.update(visible=False)),
        inputs=model_path_display,
        outputs=model_path_display
    )
    
    convert_btn.click(
        convert_to_gguf,
        inputs=outtype_dropdown,
        outputs=[gguf_file, convert_status]
    )

if __name__ == "__main__":
    os.makedirs("./finetuned_models", exist_ok=True)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
