@echo off
REM 激活 Conda 环境
call conda activate labelme

REM 设置 Labelme 路径
set LABELME_PATH=.\

REM 获取 osam 模块的路径
for /f "delims=" %%i in ('python -c "import os, osam; print(os.path.dirname(osam.__file__))"') do set OSAM_PATH=%%i

REM 运行 PyInstaller
pyinstaller __main__.py ^
  --name=Aperdata.ai.labelavm-0.5.1-se ^
  --windowed ^
  --noconfirm ^
  --specpath=./ ^
  --add-data="%OSAM_PATH%\_models\yoloworld\clip\bpe_simple_vocab_16e6.txt.gz;osam\_models\yoloworld\clip" ^
  --add-data="%LABELME_PATH%\config\default_config.yaml;labelme\config" ^
  --add-data="%LABELME_PATH%\icons\*;labelme\icons" ^
  --add-data="%LABELME_PATH%\translate\*;translate" ^
  --icon="%LABELME_PATH%\icons\icon.png" ^
  --onefile

echo Build complete!
