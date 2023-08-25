# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama



def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    The aim is to make it constantly read the cmd input and output the result.
    """

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    print("\n==================================")
    print(  "========= Start of chat ==========")
    print("==================================\n")
    while 1:
        
        cmd_input = input('> User: ')
        
        instructions = [[
            {
                "role": "user",
                "content": cmd_input,
            }
        ]]
        
        results = generator.chat_completion(
            instructions,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for instruction, result in zip(instructions, results):
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")
        
        
    


if __name__ == "__main__":
    fire.Fire(main)
