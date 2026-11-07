@echo off
rem Train va cham Verifier tren may Windows co GPU NVIDIA (vd RTX 3060 12GB).
rem Chay tu thu muc goc cua repo:
rem   lifemate\verifier\chay_verifier.bat [thu_muc_du_lieu] [backbone]
rem thu_muc_du_lieu chua runs\ver_q.jsonl, runs\dev_v4.jsonl, btc\train\... (mac dinh: ..)
rem backbone: mmbert-small (mac dinh), mmbert-base, xlmr-base, mdeberta-base
rem Ket qua: thu_muc_du_lieu\out\oracle_<backbone>.json. Du lieu BTC khong duoc commit.
setlocal
set PYTHONIOENCODING=utf-8
set DATA=%~1
if "%DATA%"=="" set DATA=..
set BB=%~2
if "%BB%"=="" set BB=mmbert-small
set BS=16
if /i not "%BB%"=="mmbert-small" set BS=8
set OUT=%DATA%\out

python -c "import torch,sys; ok=torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'KHONG CO'); sys.exit(0 if ok else 1)"
if errorlevel 1 (
  echo Chua co torch ban CUDA. Cai theo lenh tren pytorch.org/get-started ^(Windows, Pip, CUDA^).
  exit /b 1
)
python lifemate\verifier\tests\test_data_analyze.py || exit /b 1
python lifemate\verifier\train.py --data "%DATA%\runs\ver_q.jsonl" --backbone %BB% --out "%OUT%\ver_q_%BB%" --epochs 3 --bs %BS% || exit /b 1
python lifemate\verifier\oracle_ceiling.py --root "%DATA%\btc" --pred "%DATA%\runs\dev_v4.jsonl" --ids eval\splits\dev_ids.txt --model "%OUT%\ver_q_%BB%" --with-query --thresholds 0.05,0.1,0.2,0.3,0.5 --save-scores "%OUT%\p_%BB%.jsonl" --out "%OUT%\oracle_%BB%.json" || exit /b 1
echo.
echo Xong. So mean F1 tot nhat voi moc goc 0.3597. Gui file %OUT%\oracle_%BB%.json cho nhom.
