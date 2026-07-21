{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "76acd8ee",
   "metadata": {},
   "source": [
    "Nano-vLLM"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "836ca626",
   "metadata": {},
   "source": [
    "一次生成如何执行"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5415f8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    path = os.path.expanduser(\"~/huggingface/Qwen3-0.6B/\")\n",
    "    tokenizer = AutoTokenizer.from_pretrained(path)\n",
    "    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)\n",
    "\n",
    "    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)\n",
    "    prompts = [\n",
    "        \"introduce yourself\",\n",
    "        \"list all prime numbers within 100\",\n",
    "    ]\n",
    "    prompts = [\n",
    "        tokenizer.apply_chat_template(\n",
    "            [{\"role\": \"user\", \"content\": prompt}],\n",
    "            tokenize=False,\n",
    "            add_generation_prompt=True,\n",
    "        )\n",
    "        for prompt in prompts\n",
    "    ]\n",
    "    outputs = llm.generate(prompts, sampling_params)\n",
    "\n",
    "    for prompt, output in zip(prompts, outputs):\n",
    "        print(\"\\n\")\n",
    "        print(f\"Prompt: {prompt!r}\")\n",
    "        print(f\"Completion: {output['text']!r}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
