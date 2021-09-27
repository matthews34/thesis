FROM nvcr.io/nvidia/pytorch:21.04-py3

COPY . /csi_ml

# Install dependencies
WORKDIR /csi_ml
RUN conda env create --name csi_ml -f environment.yaml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "csi_ml", "/bin/bash", "-c"]

VOLUME [ "/csi_ml/output" ]

CMD [ "python", "-m" , "src"]

