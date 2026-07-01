1. Preparar o Ambiente (Em todas as VMs)
Todos os integrantes que forem rodar o projeto precisam instalar as bibliotecas de MPI, OpenMP e OpenCV no Linux. Basta rodar este comando no terminal:

sudo apt update && sudo apt install build-essential libopenmpi-dev openmpi-bin libopencv-dev pkg-config

2. Como Compilar o Código
Salvem o código C em um arquivo chamado trabalho_ppd.c. Para compilar, vocês não usarão o gcc normal, mas sim o compilador do MPI, passando a flag do OpenMP e linkando o OpenCV:

mpic++ -fopenmp trabalho_ppd.c -o trabalhador_video $(pkg-config --cflags --libs opencv4)

Se o comando acima der erro de pacote não encontrado, tentem trocar opencv4 apenas por opencv.

3. Como Rodar Localmente (Para testes rápidos em 1 VM)
Antes de testar na rede, verifiquem se funciona na própria máquina. O comando abaixo inicia 4 processos (1 Coordenador e 3 Trabalhadores) no mesmo computador:

mpirun -np 4 ./trabalhador_video

4. Como Rodar Distribuído (O objetivo final do trabalho)
Para rodar dividindo o processamento entre as várias VMs do VirtualBox, vocês precisam criar um arquivo de texto chamado hosts.txt contendo os IPs das máquinas virtuais do grupo. Depois, executem:

mpirun -np 4 --hostfile hosts.txt ./trabalhador_video
