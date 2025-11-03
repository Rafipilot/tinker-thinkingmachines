import logging
import time

import chz
import datasets
import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from datasets import load

import math
import os 
from config import tinker_api_key
os.environ["TINKER_API_KEY"] = tinker_api_key

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/sl-loop"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 128
    learning_rate: float = 1e-4
    max_length: int = 32768
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    lora_rank: int = 32
    save_every: int = 20


def main(config: Config):
    ml_logger = ml_log.setup_logging(
    log_dir=config.log_path,
    wandb_project=None,
    wandb_name=None,
    config=config,
    do_configure_logging_module=True,
    )
    tokenizer = get_tokenizer(config.model_name)
    
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    dataset = datasets.load_dataset("json", data_files={"train":"example_data.jsonl"})
    train_dataset = dataset["train"]

    n_train_batches = math.ceil(len(train_dataset) / config.batch_size)
    
    service_client = tinker.ServiceClient(base_url = config.base_url)

    resume_info = False # scheckpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    train_dataset = train_dataset.shuffle(seed=0)

    print("Starting training, num batches: ", n_train_batches)
    for i in range(6): # since my custom dataset is tiny im doing 6 epochs
        for batch_idx in range (start_batch, n_train_batches):
            start_time = time.time()
            step = batch_idx
            metrics = {}

            lr_mult = max(0, 1-step/n_train_batches)
            current_lr = config.learning_rate * lr_mult
            adam_params = tinker.AdamParams (learning_rate=current_lr, beta1 = 0.9, beta2=0.95, eps=1e-8)

            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
            batch_rows = train_dataset.select(range(batch_start, batch_end))

            batch = [
                conversation_to_datum(
                    row["messages"],  # type: ignore
                    renderer,
                    config.max_length,
                    config.train_on_what,
                )
                for row in batch_rows
            ]

            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_step_future = training_client.optim_step(adam_params)

                    # Training step
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_step_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            _optim_result = optim_step_future.result()

            # Compute train metrics
            train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            # Log metrics
            metrics.update(
                num_sequences=len(batch),
                num_tokens=sum(d.model_input.length for d in batch),
                learning_rate=current_lr,
                train_mean_nll=train_nll,
                progress=step / n_train_batches,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=step)

    # Save final checkpoint
    sampling_client = training_client.save_weights_and_get_sampling_client(name="Rafi-test04")

    message = [{"role": "user", "content": "What are Newton’s three laws of motion?"
    }]
    prompt = renderer.build_generation_prompt(message) # hopefully this will return the prompt
    
    sampling_params = types.SamplingParams(
        max_tokens=50,
        temperature=0.7,
        top_p=0.95,

    )

    future = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1)

    result = future.result()
    for seq in result.sequences:
        text = tokenizer.decode(seq.tokens)
        print("Text: ", text)

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    return training_client, tokenizer

if __name__ == "__main__":
    chz.nested_entrypoint(main)


#tinker_cookbook.checkpoint_utils:75 [INFO] Saved checkpoints: {'state_path': 'tinker://7c7cb23a-775a-4da8-9978-a2c4ed4ee310/weights/final', 'sampler_path': 'tinker://7c7cb23a-775a-4da8-9978-a2c4ed4ee310/sampler_weights/final'}