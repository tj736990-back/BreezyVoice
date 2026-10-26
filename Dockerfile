FROM eugr/spark-vllm-b12x:latest
WORKDIR /breezyvoice

ENV UV_LINK_MODE=copy

# base image already has torch 2.13.0+cu130 / torchaudio 2.11.0+cu130 (sm_121 support)
# add missing build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --break-system-packages "setuptools<81" wheel "Cython<3"

RUN wget -q -O /opt/openfst-1.8.2.tar.gz https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.2.tar.gz && \
    tar xzf /opt/openfst-1.8.2.tar.gz -C /opt && \
    cd /opt/openfst-1.8.2 && \
    ./configure --enable-grm --prefix=/usr/local && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /opt/openfst-1.8.2 /opt/openfst-1.8.2.tar.gz

RUN uv pip install --system --break-system-packages --no-build-isolation pynini==2.1.5

COPY requirements.txt /breezyvoice/requirements.txt
COPY build-constraints.txt /breezyvoice/build-constraints.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -r requirements.txt --build-constraint build-constraints.txt

COPY . .

# torchaudio>=2.1 removed set_audio_backend; guard the legacy call
RUN python3 - <<'PY'
import pathlib
p = pathlib.Path("/breezyvoice/cosyvoice/dataset/processor.py")
s = p.read_text(encoding="utf-8")
old = "torchaudio.set_audio_backend('soundfile')\n"
new = "try:\n    torchaudio.set_audio_backend('soundfile')\nexcept AttributeError:\n    pass\n"
assert old in s, "set_audio_backend pattern not found"
p.write_text(s.replace(old, new), encoding="utf-8")
print("processor.py patched (guard set_audio_backend)")
PY

RUN uv pip install --system --break-system-packages --no-build-isolation --no-deps ./third_party/Matcha-TTS

RUN python3 - <<'PY'
import pathlib, g2pw
p = pathlib.Path(g2pw.__file__).parent / "api.py"
s = p.read_text(encoding="utf-8")
old = "self.session_g2pw =  onnxruntime.InferenceSession(onnx_path, sess_options=sess_options)"
new = "self.session_g2pw =  onnxruntime.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])"
assert old in s, "g2pw CPU-session pattern not found"
p.write_text(s.replace(old, new), encoding="utf-8")
print("g2pw api.py patched (explicit CPUExecutionProvider)")
PY

RUN python3 - <<'PY'
import urllib.request, zipfile, shutil, os
url = "https://huggingface.co/YC-Chen/g2pW/resolve/main/G2PWModel-v2-onnx.zip"
tmp = "/tmp/g2pw.zip"
dest = "/breezyvoice/G2PWModel"
urllib.request.urlretrieve(url, tmp)
z = zipfile.ZipFile(tmp)
top = z.namelist()[0].split("/")[0]
z.extractall("/tmp/g2pw_extract")
z.close()
if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.move("/tmp/g2pw_extract/" + top, dest)
os.remove(tmp)
shutil.rmtree("/tmp/g2pw_extract")
print("G2PW model pre-baked at", dest)
PY

RUN HF_HOME=/breezyvoice/.hf-cache python3 - <<'PY'
from transformers import BertTokenizer
BertTokenizer.from_pretrained("bert-base-chinese")
print("bert-base-chinese tokenizer pre-baked")
PY

EXPOSE 8080

ENTRYPOINT ["/usr/bin/python3"]
