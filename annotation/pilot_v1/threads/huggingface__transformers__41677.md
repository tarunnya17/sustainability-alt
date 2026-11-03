# #41677 [pull_request] Remove upper version bound of pandas
https://github.com/huggingface/transformers/pull/41677  ·  opened by cyyever on 2025-10-17

## [body] cyyever (CONTRIBUTOR) · opening post · 2025-10-17
# What does this PR do?

Let's skip pandas 2.3.0 and allow users to install newer versions.

## [c3415484588] Rocketknight1 (MEMBER) · issue_comment · 2025-10-17
cc @ydshieh since I don't know why that version skip was added!

## [c3415953598] ydshieh (COLLABORATOR) · issue_comment · 2025-10-17
It is #38605. Maybe it works now but I don't want to spend the time to check. 

PR LGTM, thank you

The error at that time:

> Using Python 3.9.23 environment at: /usr/local
Resolved 47 packages in 3.41s
      Built transformers @ file:///root/transformers
  × Failed to build `pandas==2.3.0`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `mesonpy.build_wheel` failed (exit status: 1)

      [stdout]
      + meson setup
      /root/.cache/uv/sdists-v9/pypi/pandas/2.3.0/VqR3nkX0wjeGwwtaucpZ_/src
      /root/.cache/uv/sdists-v9/pypi/pandas/2.3.0/VqR3nkX0wjeGwwtaucpZ_/src/.mesonpy-2kp275i8
      -Dbuildtype=release -Db_ndebug=if-release -Db_vscrt=md --vsenv
      --native-file=/root/.cache/uv/sdists-v9/pypi/pandas/2.3.0/VqR3nkX0wjeGwwtaucpZ_/src/.mesonpy-2kp275i8/meson-python-native-file.ini
      The Meson build system
      Version: 1.8.1
      Source dir:
      /root/.cache/uv/sdists-v9/pypi/pandas/2.3.0/VqR3nkX0wjeGwwtaucpZ_/src
      Build dir:
      /root/.cache/uv/sdists-v9/pypi/pandas/2.3.0/VqR3nkX0wjeGwwtaucpZ_/src/.mesonpy-2kp275i8
      Build type: native build
      Project name: pandas
      Project version: 2.3.0

      ../meson.build:2:0: ERROR: Unknown compiler(s): [['cc'], ['gcc'],
      ['clang'], ['nvc'], ['pgcc'], ['icc'], ['icx']]
      The following exception(s) were encountered:
      Running `cc --version` gave "[Errno 2] No such file or directory: 'cc'"
      Running `gcc --version` gave "[Errno 2] No such file or directory:
      'gcc'"
      Running `clang --version` gave "[Errno 2] No such file or directory:
      'clang'"
      Running `nvc --version` gave "[Errno 2] No such file or directory:
      'nvc'"
      Running `pgcc --version` gave "[Errno 2] No such file or directory:
      'pgcc'"
      Running `icc --version` gave "[Errno 2] No such file or directory:
      'icc'"
      Running `icx --version` gave "[Errno 2] No such file or directory:
      'icx'"

      A full log can be found at
      /root/.cache/uv/sdists-v9/pypi/pandas/2.3.0/VqR3nkX0wjeGwwtaucpZ_/src/.mesonpy-2kp275i8/meson-logs/meson-log.txt

      hint: This usually indicates a problem with the package or the build
      environment.
  help: `pandas` (v2.3.0) was included because `transformers[quality]`
        (v4.53.0.dev0) depends on `datasets` (v3.6.0) which depends on
        `pandas`

Exited with code exit status 1

## [c3415976536] ydshieh (COLLABORATOR) · issue_comment · 2025-10-17
CI is already on 

pandas==2.3.3

so we are good

## [c3350860601] ydshieh (COLLABORATOR) · review APPROVED · 2025-10-17
💯

---
outcome: state=closed, merged=True
