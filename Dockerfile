FROM pytorch/pytorch

RUN apt-get update && \
    apt-get install -y texlive-science texlive-latex-extra dvipng && \
    rm -rf /var/lib/apt/lists/*

COPY . /home/root/BayesCBF
WORKDIR /home/root/BayesCBF
RUN pip install --no-cache-dir .
RUN python setup.py test
