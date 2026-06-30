#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Headers da API C Clássica do OpenCV */
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio/videoio_c.h>

/* Função de Filtro Manual (Box Blur 3x3) utilizando OpenMP */
void applyManualFilter(const unsigned char* input, unsigned char* output, int width, int height, int channels, int widthStep) {
    /* Copia as bordas para não processar memória fora dos limites */
    memcpy(output, input, height * widthStep);

    /* Paraleliza o processamento das linhas do frame usando OpenMP */
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                int sum = 0;
                /* Aplica a máscara 3x3 manualmente acessando os pixels */
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        /* Cálculo do índice linear usando widthStep para garantir o alinhamento correto da memória */
                        int pixel_idx = (y + ky) * widthStep + (x + kx) * channels + c;
                        sum += input[pixel_idx];
                    }
                }
                int out_idx = y * widthStep + x * channels + c;
                output[out_idx] = sum / 9; /* Média dos 9 pixels */
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    int provided;

    /* Inicializa MPI com suporte a threads */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("Erro: O programa requer pelo menos 2 processos (1 Coordenador + 1 Trabalhador).\n");
        }
        MPI_Finalize();
        return -1;
    }

    /* Array para enviar dimensões: [0]=width, [1]=height, [2]=channels, [3]=widthStep */
    int frame_info[4]; 
    int stop_signal = 0;

    if (rank == 0) {
        /* ========================================== */
        /* PROCESSO COORDENADOR                       */
        /* ========================================== */
        
        CvCapture* capture = cvCreateCameraCapture(0); /* Mude para cvCreateFileCapture("video.mp4") para arquivo */
        if (!capture) {
            printf("Erro ao abrir a câmera/vídeo.\n");
            stop_signal = 1;
        }

        /* Envia sinal de aborto caso a captura falhe */
        for (int i = 1; i < size; i++) {
            MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        if (stop_signal) {
            MPI_Finalize();
            return -1;
        }

        IplImage* frame = cvQueryFrame(capture);
        if (!frame) return -1;

        frame_info[0] = frame->width;
        frame_info[1] = frame->height;
        frame_info[2] = frame->nChannels;
        frame_info[3] = frame->widthStep; /* Importante na API C para pular linhas corretamente */
        long data_size = frame->height * frame->widthStep;

        /* Envia as dimensões para os trabalhadores */
        for (int i = 1; i < size; i++) {
            MPI_Send(frame_info, 4, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

        /* Aloca array de imagens para receber os resultados */
        IplImage** frames_received = (IplImage**)malloc((size - 1) * sizeof(IplImage*));
        for (int i = 0; i < size - 1; i++) {
            frames_received[i] = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, frame->nChannels);
        }

        cvNamedWindow("Video Processado (MPI + OpenMP)", CV_WINDOW_AUTOSIZE);

        int playing = 1;
        while (playing) {
            int workers_active = 0;

            /* 1. Coordenador organiza e envia frames */
            for (int i = 1; i < size; i++) {
                frame = cvQueryFrame(capture);
                if (!frame) {
                    playing = 0;
                    break;
                }
                
                MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(frame->imageData, data_size, MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD);
                workers_active++;
            }

            /* 2. Coordenador recebe e exibe os frames processados */
            for (int i = 1; i <= workers_active; i++) {
                MPI_Recv(frames_received[i-1]->imageData, data_size, MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                cvShowImage("Video Processado (MPI + OpenMP)", frames_received[i-1]);
                if (cvWaitKey(1) == 27) { /* ESC para sair */
                    playing = 0;
                }
            }
        }

        /* Avisa os trabalhadores para pararem */
        stop_signal = 1;
        for (int i = 1; i < size; i++) {
            MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        /* Limpeza de memória do mestre */
        cvReleaseCapture(&capture);
        cvDestroyAllWindows();
        for (int i = 0; i < size - 1; i++) {
            cvReleaseImage(&frames_received[i]);
        }
        free(frames_received);

    } else {
        /* ========================================== */
        /* PROCESSOS TRABALHADORES                    */
        /* ========================================== */
        
        MPI_Recv(&stop_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (stop_signal) {
            MPI_Finalize();
            return 0;
        }

        MPI_Recv(frame_info, 4, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int width = frame_info[0];
        int height = frame_info[1];
        int channels = frame_info[2];
        int widthStep = frame_info[3];
        long data_size = height * widthStep;

        /* Alocação manual usando C */
        unsigned char* input_buffer = (unsigned char*)malloc(data_size * sizeof(unsigned char));
        unsigned char* output_buffer = (unsigned char*)malloc(data_size * sizeof(unsigned char));

        while (1) {
            MPI_Recv(&stop_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (stop_signal) break;

            /* Trabalhador recebe o frame */
            MPI_Recv(input_buffer, data_size, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Trabalhador processa localmente os pixels utilizando OpenMP */
            applyManualFilter(input_buffer, output_buffer, width, height, channels, widthStep);

            /* Trabalhador retorna o frame */
            MPI_Send(output_buffer, data_size, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
        }

        free(input_buffer);
        free(output_buffer);
    }

    MPI_Finalize();
    return 0;
}