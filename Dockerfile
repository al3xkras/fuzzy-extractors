FROM python:3.10

RUN apt update && apt-get install -y git libsasl2-dev python-dev libldap2-dev libssl-dev openssh-server

RUN mkdir fuzzy && chmod a+rwx /fuzzy

RUN pip install cmake

RUN pip install dlib

COPY ./python ./fuzzy
RUN cd fuzzy && python3 venv_setup.py

RUN apt update && apt -y install openssh-server whois

ARG USERNAME=sshuser
ARG USERPASS=sshpass

RUN useradd -ou 0 -g 0 -ms /bin/bash $USERNAME
RUN usermod --password $(echo "$USERPASS" | mkpasswd -s) $USERNAME

RUN apt purge -y whois && apt -y autoremove && apt -y autoclean && apt -y clean

USER $USERNAME
RUN mkdir /home/$USERNAME/.ssh && touch /home/$USERNAME/.ssh/authorized_keys
USER root

VOLUME /home/$USERNAME/.ssh
VOLUME /etc/ssh

COPY ./sshd_config ./etc/ssh/sshd_config

CMD service ssh start && bash