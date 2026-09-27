# faster-whisper-GUI

    faster-whisper、whisperX，GUI with PySide6

- ## Installation (uv-managed)

  Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`
  (this replaces the old `requirements.txt`, which had drifted out of date).

  - **Prerequisites**
    - [uv](https://docs.astral.sh/uv/) — `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
    - AMD HIP SDK (default `C:\Program Files\AMD\ROCm\<ver>`)
    - Intel oneAPI (provides `dnnl` for the ROCm CTranslate2 build)

  - **Python versions**
    - `3.13` is the default (pinned in `.python-version`). No dependency changes
      are needed for it.
    - `3.12` also works — it is the version this project was originally
      validated and benchmarked on.
    - `3.14` works, but PyAudio ships no `cp314` wheel, so the
      [PyAudioWPatch](https://pypi.org/project/PyAudioWPatch/) fork is used
      automatically instead (see the import fallback in
      `faster_whisper_GUI/transcribe.py`).
    - `3.15` is not usable yet — `ctranslate2`, `torch` and `onnxruntime` have no
      3.15 wheels at all, and AMD's ROCm torch stops at `cp314`.
    - To switch: `uv sync --extra rocm --python 3.12`, then re-run `setup.ps1`
      to replace the CTranslate2 DLL.
    - Note: all three supported versions transcribe at the same speed (measured
      8.14x / 7.85x / 7.98x realtime — within run-to-run noise). The version
      choice is about interpreter currency, not performance.

  - **Install**
    ```powershell
    powershell -ExecutionPolicy Bypass -File setup.ps1
    ```
    This runs `uv sync --extra rocm`, drops the locally built ROCm
    `ctranslate2.dll` into site-packages, and verifies that the GPU is visible.

  - **Run**
    ```powershell
    .\启动GUI.bat
    # or
    uv run FasterWhisperGUI.py
    ```

  - **Changing dependencies** — edit `pyproject.toml`, then `uv lock`.
    Note that `uv sync` is exact: reinstalling `ctranslate2` overwrites the
    custom ROCm DLL, so re-run `setup.ps1` afterwards to restore it.

  > The ROCm `ctranslate2.dll` is built from CTranslate2 `v4.8.2` source and must
  > stay version-matched with the `ctranslate2` package. Do not bump
  > `ctranslate2` without rebuilding the DLL — see `.probe/FEASIBILITY.md`.

- ## model download

  - https://huggingface.co/models?sort=trending&search=faster-whisper
  
  - you can also download and convert models in software

  - large-v3 model float32 :
  
    - [Huggingface](https://huggingface.co/CheshireCC/faster-whisper-large-v3-float32)
    
    - [百度云网盘链接](https://pan.baidu.com/s/1qltCehSq3pWMlIJ06sWLCQ?pwd=5xq8)
    
        
  
- ### Links

  - [pyside6-fluent-widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
  - [faster-whisper](https://github.com/guillaumekln/faster-whisper)
  - [whisperX](https://github.com/m-bain/whisperX)
  - [HuggingFace models download](https://huggingface.co/models)
  - [Demucs](https://github.com/facebookresearch/demucs)
  - more and better AVE ：

    - [UVR](https://github.com/Anjok07/ultimatevocalremovergui#installation)
    - [Demucs-Gui](https://github.com/CarlGao4/Demucs-Gui)
  
- ## What's this

  - this is a GUI software of faster-whisper , you can:
    - Transcrib audio or video files to srt/txt/smi/vtt/lrc file
    - provide all paraments of VAD-model and whisper-model
    - now, it support whisperX
    - Demucs model support
    - whisper large-v3 model support

---

- ## Best wishes to the world that received this message

  - ### Agreement

    - By using this software, you have read and agreed to the following user agreement:
      - You agree to use this software in compliance with the laws of your country or region.
      - You may not perform, including, but not limited to, the following acts, nor facilitate any violation of the law:
        - those who oppose the basic principles laid down in the Constitution.
        - endangering national security, divulging state secrets, subverting state power and undermining national unity.
        - harming the honor and interests of the country.
        - inciting ethnic hatred and racial discrimination.
        - those who sabotage the country's religious policy and promote cults.
        - spreading rumors, disturbing social order and undermining social stability.
        - spreading pornography, gambling, violence, murder, terrorism or abetting crime.
        - insulting or slandering others and infringing upon the legitimate rights and interests of others.
        - containing other contents prohibited by laws or administrative regulations.
      - All consequences and responsibilities caused by violations of laws and regulations in any related matters such as the generation, collection, processing and use of your data shall be borne by you.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=CheshireCC/faster-whisper-GUI&type=Timeline)](https://star-history.com/#CheshireCC/faster-whisper-GUI&Timeline)

- ### UI Language ###

    ![屏幕截图 2024-03-11 183130](./README.assets/183130.png)

- ### Theme Color ###

    ![屏幕截图 2024-03-11 184459](./README.assets/184459.png)

    ![image-20240311184818398](./README.assets/image-20240311184818398.png)

- ### Load Model / Download Model / Convert Model

![image-20231118155123131](./README.assets/image-20231118155123131.png)

- ### Large-v3 模型支持

  ![image-20231118155209847](./README.assets/image-20231118155209847.png)
- ### Demucs AVE

  ![DemucsFunction](./README.assets/DemucsFunction.png)
- ### batch process

![image-20231008150849827](./README.assets/image-20231008150849827.png)

- ### File List

  ![0.3.0_newFIleSystem](./README.assets/0.3.0_newFIleSystem.png)
- ### FileFilter

  ![fileFilter](./README.assets/fileFilter.png)
- ### WhisperX function

![0.3.0_whisperx](./README.assets/0.3.0_whisperx.png)

- ### paraments of faster-whisper model

![image-20231113020210745](./README.assets/image-20231113020210745.png)

- ### Silero VAD

  ![image-20231113020407272](./README.assets/image-20231113020407272.png)
- ### setting

  ![image-20231118155300816](./README.assets/image-20231118155300816.png)
- ### Show result and edit timestample

  ![0.3.0_result](./README.assets/0.3.0_result.png)![image-20231007191942864](./README.assets/image-20231007191942864.png)
- ### words-level timestamps —— karaoka lyric (work in `VTT`/`LRC`/`SMI` format)


  - play with foobar2000 , ESLyric plugin, `lrc` format lyric

  ![image-20230811130449688](./README.assets/image-20230811130449688.png)
