"""
Expert-Balanced Self-Sampling (EBSS)

Beam-based self-sampling that generates calibration data with balanced expert activation.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class EBSSConfig:
    """Configuration for EBSS sampling"""
    beam_width: int = 4
    tau: float = 1.2  # Temperature for expert balance scoring
    max_tokens: int = 512
    num_samples: int = 100
    min_perplexity: float = 1.0
    max_perplexity: float = 100.0
    expert_balance_weight: float = 1.0


class EBSSSampler:
    """
    Expert-Balanced Self-Sampling

    Generates calibration data by beam search with scoring:
    score = perplexity_term + (sigma_expert_balance / tau)

    This ensures generated samples cover diverse expert activation patterns.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[EBSSConfig] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EBSSConfig()
        self.device = device
        self.expert_counts = {}  # Track expert activation counts

    def reset_expert_counts(self):
        """Reset expert activation counters"""
        self.expert_counts = {}

    def update_expert_counts(self, expert_ids: torch.Tensor):
        """Update expert activation counts from routing decisions"""
        for eid in expert_ids.flatten().tolist():
            self.expert_counts[eid] = self.expert_counts.get(eid, 0) + 1

    def compute_expert_balance_score(self) -> float:
        """
        Compute expert balance metric (lower is more balanced)

        Returns standard deviation of expert activation counts
        """
        if not self.expert_counts:
            return 0.0
        counts = list(self.expert_counts.values())
        return float(np.std(counts))

    def compute_perplexity(self, logits: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Compute perplexity for generated sequence"""
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1,
                                            target_ids.unsqueeze(-1)).squeeze(-1)
        nll = -target_log_probs.mean()
        ppl = torch.exp(nll).item()
        return np.clip(ppl, self.config.min_perplexity, self.config.max_perplexity)

    def compute_beam_score(self, perplexity: float, expert_balance: float) -> float:
        """
        Compute overall beam score

        score = perplexity + (expert_balance / tau)
        Lower is better (we want low perplexity and balanced experts)
        """
        return perplexity + (expert_balance / self.config.tau)

    @torch.no_grad()
    def generate(
        self,
        seed_texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Generate calibration samples using EBSS

        Args:
            seed_texts: Initial prompts to start generation
            progress_callback: Optional callback(current, total)

        Returns:
            List of generated text samples with balanced expert coverage
        """
        self.model.eval()
        generated_samples = []

        for idx, seed_text in enumerate(seed_texts):
            if progress_callback:
                progress_callback(idx, len(seed_texts))

            # Reset expert counts for this sample
            self.reset_expert_counts()

            # Tokenize seed
            input_ids = self.tokenizer.encode(
                seed_text, return_tensors="pt").to(self.device)

            # Initialize beams: [(token_ids, score, expert_counts)]
            beams = [(input_ids[0], 0.0, {})]

            # Beam search
            for step in range(self.config.max_tokens):
                all_candidates = []

                for beam_ids, beam_score, beam_expert_counts in beams:
                    # Prepare input
                    input_tensor = beam_ids.unsqueeze(0)

                    # Forward pass (with expert tracking hook if available)
                    outputs = self.model(
                        input_tensor, use_cache=False, return_dict=True)
                    logits = outputs.logits[0, -1, :]  # Last token logits

                    # Get expert IDs if available
                    if hasattr(outputs, "expert_ids") and outputs.expert_ids is not None:
                        expert_ids = outputs.expert_ids
                        self.update_expert_counts(expert_ids)

                    # Sample top-k next tokens
                    top_k_probs, top_k_ids = torch.topk(
                        F.softmax(logits, dim=-1),
                        k=min(self.config.beam_width, logits.size(-1))
                    )

                    # Create candidates
                    for prob, next_id in zip(top_k_probs, top_k_ids):
                        new_ids = torch.cat([beam_ids, next_id.unsqueeze(0)])

                        # Compute perplexity (approximate from log prob)
                        token_ppl = 1.0 / (prob.item() + 1e-10)

                        # Compute expert balance
                        expert_balance = self.compute_expert_balance_score()

                        # Compute total score
                        new_score = self.compute_beam_score(
                            token_ppl, expert_balance)

                        all_candidates.append(
                            (new_ids, new_score, dict(self.expert_counts)))

                # Select top-w beams
                beams = sorted(all_candidates, key=lambda x: x[1])[
                    :self.config.beam_width]

                # Check for EOS
                if all(self.tokenizer.eos_token_id in beam[0] for beam in beams):
                    break

            # Take best beam
            best_ids = beams[0][0]
            generated_text = self.tokenizer.decode(
                best_ids, skip_special_tokens=True)
            generated_samples.append(generated_text)

        return generated_samples

    def generate_batch(
        self,
        seed_texts: List[str],
        batch_size: int = 4
    ) -> List[str]:
        """Generate samples in batches for efficiency"""
        all_samples = []
        for i in range(0, len(seed_texts), batch_size):
            batch = seed_texts[i:i+batch_size]
            samples = self.generate(batch)
            all_samples.extend(samples)
        return all_samples


def create_ebss_sampler(
    model,
    tokenizer,
    beam_width: int = 4,
    tau: float = 1.2,
    max_tokens: int = 512,
    device: str = "cuda"
) -> EBSSSampler:
    """Convenience function to create EBSS sampler"""
    config = EBSSConfig(
        beam_width=beam_width,
        tau=tau,
        max_tokens=max_tokens
    )
    return EBSSSampler(model, tokenizer, config, device)
