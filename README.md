# AprendizadoDeMaquina

Para criar o ambiente virtual, abra o terminal dentro da pasta e faça:
python3 -m venv ambiente_virtual

Mas, não basta somente criar o ambiente, é necessário ativá-lo:
source ambiente_virtual/bin/activate

Agora, se quisermos rodar o nosso projeto em outra máquina, não será necessário baixar as dependências uma a uma, basta fazer:

pip install -r requirements.txt

E por fim, para desativar o ambiente virtual:

deactivate
