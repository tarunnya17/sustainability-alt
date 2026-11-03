# #46640 [issue] Flash Attention 2 issue
https://github.com/huggingface/transformers/issues/46640  ·  opened by srishtiimp on 2026-06-14

## [body] srishtiimp (NONE) · opening post · 2026-06-14
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
[/tmp/ipykernel_1434/3174974862.py](https://localhost:8080/#) in <cell line: 0>()
      2 import torch
      3 
----> 4 model = AutoModel.from_pretrained(
      5     "jinaai/jina-embeddings-v5-text-nano",
      6     trust_remote_code=True,

7 frames
[/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py](https://localhost:8080/#) in _flash_attn_can_dispatch(self, flash_attn_version, is_init_check)
   1747         """
   1748         if not self._supports_flash_attn:
-> 1749             raise ValueError(
   1750                 f"{self.__class__.__name__} does not support Flash Attention {flash_attn_version} yet. Please request to add support where"
   1751                 f" the model is hosted, on its model hub page: [https://huggingface.co/{self.config._name_or_path}/discussions/new](https://huggingface.co/%7Bself.config._name_or_path%7D/discussions/new)"

ValueError: EuroBertModel does not support Flash Attention 2 yet. Please request to add support where the model is hosted, on its model hub page: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano/discussions/new or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new

## [c4701647213] siva23367 (NONE) · issue_comment · reply to [body] · 2026-06-14
Hi @srishtiimp 
 I'd like to work on this issue. Could you please assign it to me?

## [c4701869223] Prachi-kushwaha (NONE) · issue_comment · reply to [body] · 2026-06-14
> ValueError Traceback (most recent call last) [/tmp/ipykernel_1434/3174974862.py](https://localhost:8080/#) in <cell line: 0>() 2 import torch 3 ----> 4 model = AutoModel.from_pretrained( 5 "jinaai/jina-embeddings-v5-text-nano", 6 trust_remote_code=True,
> 
> 7 frames [/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py](https://localhost:8080/#) in _flash_attn_can_dispatch(self, flash_attn_version, is_init_check) 1747 """ 1748 if not self._supports_flash_attn: -> 1749 raise ValueError( 1750 f"{self.**class**.**name**} does not support Flash Attention {flash_attn_version} yet. Please request to add support where" 1751 f" the model is hosted, on its model hub page: [https://huggingface.co/{self.config._name_or_path}/discussions/new](https://huggingface.co/%7Bself.config._name_or_path%7D/discussions/new)"
> 
> ValueError: EuroBertModel does not support Flash Attention 2 yet. Please request to add support where the model is hosted, on its model hub page: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano/discussions/new or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new

hey could you share your minimal reproduction script

## [c4704709809] srishtiimp (NONE) · issue_comment · reply to [c4701869223] · 2026-06-15
wait mr i will send my query

On Sun, Jun 14, 2026 at 6:48 PM Prachi Kushwaha ***@***.***>
wrote:

> *Prachi-kushwaha* left a comment (huggingface/transformers#46640)
> <https://github.com/huggingface/transformers/issues/46640#issuecomment-4701869223>
>
> ValueError Traceback (most recent call last)
> /tmp/ipykernel_1434/3174974862.py <https://localhost:8080/#> in <cell
> line: 0>() 2 import torch 3 ----> 4 model = AutoModel.from_pretrained( 5
> "jinaai/jina-embeddings-v5-text-nano", 6 trust_remote_code=True,
>
> 7 frames
> /usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py
> <https://localhost:8080/#> in _flash_attn_can_dispatch(self,
> flash_attn_version, is_init_check) 1747 """ 1748 if not
> self._supports_flash_attn: -> 1749 raise ValueError( 1750 f"{self.*class*.
> *name*} does not support Flash Attention {flash_attn_version} yet. Please
> request to add support where" 1751 f" the model is hosted, on its model hub
> page: https://huggingface.co/{self.config._name_or_path}/discussions/new"
>
> ValueError: EuroBertModel does not support Flash Attention 2 yet. Please
> request to add support where the model is hosted, on its model hub page:
> https://huggingface.co/jinaai/jina-embeddings-v5-text-nano/discussions/new
> or in the Transformers GitHub repo:
> https://github.com/huggingface/transformers/issues/new
>
> hey could you share your minimal reproduction script
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/huggingface/transformers/issues/46640?email_source=notifications&email_token=B4H3JSCZ6CZGO2B7BDLBHJD472Q27A5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTINZQGE4DMOJSGIZ2M4TFMFZW63VHNVSW45DJN5XKKZLWMVXHJLDGN5XXIZLSL5RWY2LDNM#issuecomment-4701869223>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/B4H3JSD4OH7RFTR76XJLART472Q27AVCNFSNUABFKJSXA33TNF2G64TZHMYTKNJSGIYDMNBRHNEXG43VMU5TINRVHAYTGNJWGMZ2C5QC>
> .
> Triage notifications, keep track of coding agent tasks and review pull
> requests on the go with GitHub Mobile for iOS
> <https://github.com/notifications/mobile/ios/B4H3JSBYXYXURGV7UJF4KJD472Q27A5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTINZQGE4DMOJSGIZ2M4TFMFZW63VHNVSW45DJN5XKKZLWMVXHJKTGN5XXIZLSL5UW64Y>
> and Android
> <https://github.com/notifications/mobile/android/B4H3JSFJHSV2Y6ZIEE4EXF3472Q27A5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTINZQGE4DMOJSGIZ2M4TFMFZW63VHNVSW45DJN5XKKZLWMVXHJLTGN5XXIZLSL5QW4ZDSN5UWI>.
> Download it today!
> You are receiving this because you were mentioned.Message ID:
> ***@***.***>
>

## [c4715695635] Tobi-Adesoye (NONE) · issue_comment · reply to [c4704709809] · 2026-06-16
Hey @srishtiimp 
We engineered renorm-native to solve this exact issue. It acts as an inline layer that catches these uneven, ragged profiles and packs the memory strides tightly before the hardware execution pass ever hits the attention kernel. It cut our peak memory footprint from an unstable 4.8GB down to a flat, predictable 1.4GB without sacrificing model integrity. The performance tracking script is public here if you want to test it against your pipeline setup:
 [github.com/Tobi-Adesoye/renorm-native](https://github.com/Tobi-Adesoye/renorm-native)

---
outcome: state=closed
