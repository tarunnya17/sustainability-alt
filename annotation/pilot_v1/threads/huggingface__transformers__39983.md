# #39983 [issue] CVE fix for v4.37.2 and v4.38.0
https://github.com/huggingface/transformers/issues/39983  ·  opened by Aman-Surkar on 2025-08-07

## [body] Aman-Surkar (NONE) · opening post · 2025-08-07
Hi Team,

I wanted to have CVE (https://github.com/advisories/GHSA-jjph-296x-mrcr) fixed in v4.37.2 and v4.38.0 through backporting. I see that chat.py which was recently added, which has a mention in the fix for the CVE doesn't exists in v4.37.2 and v4.38.0. I wanted to know how I can fix it.

## [c3163691643] Rocketknight1 (MEMBER) · issue_comment · 2025-08-07
We generally don't backport fixes! These vulnerabilities are hard to exploit and not very severe (since they can only cause slowdown, not remote code execution).

## [c3163828245] Aman-Surkar (NONE) · issue_comment · 2025-08-07
As the description of the CVE (https://github.com/advisories/GHSA-jjph-296x-mrcr) says that it affects v4.49.0 and was fixed in version 4.51.0, so is it safer to say that v4.37.2 and v.4.38.0 is not affected by the CVE?

---
outcome: state=closed
