#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Usando a API Moderna em C++ do OpenCV */
#include <opencv2/opencv.hpp>

using namespace cv;

/* Função de Filtro Manual (Convolução 7x7) utilizando OpenMP */
void applyManualFilter(const unsigned char* input, unsigned char* output, int width, int height, int channels, int widthStep, int rank, int frame_id) {
    /* Copia as bordas para não processar memória fora dos limites */
    memcpy(output, input, height * widthStep);

    int kernel_size = 7; 
    int k_half = kernel_size / 2;

    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("[Trabalhador %d] Processando frame %d com %d threads (OpenMP)...\n", 
                   rank, frame_id, omp_get_num_threads());
        }

        #pragma omp for
        for (int y = k_half; y < height - k_half; y++) {
            for (int x = k_half; x < width - k_half; x++) {
                for (int c = 0; c < channels; c++) {
                    int sum = 0;
                    for (int ky = -k_half; ky <= k_half; ky++) {
                        for (int kx = -k_half; kx <= k_half; kx++) {
                            int pixel_idx = (y + ky) * widthStep + (x + kx) * channels + c;
                            sum += input[pixel_idx];
                        }
                    }
                    int out_idx = y * widthStep + x * channels + c;
                    output[out_idx] = sum / (kernel_size * kernel_size); 
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, size, provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Erro: O programa requer pelo menos 2 processos.\n");
        MPI_Finalize();
        return -1;
    }

    int frame_info[4]; 
    int stop_signal = 0;

    if (rank == 0) {
        /* ========================================== */
        /* PROCESSO COORDENADOR                       */
        /* ========================================== */
        printf("[Coordenador] Iniciando sistema com %d processos.\n", size);
        
        /* Lê a partir do arquivo de vídeo */
        VideoCapture capture("video.mp4"); 
        
        if (!capture.isOpened()) {
            printf("[Coordenador] Erro ao abrir a camera/video. Verifique se o arquivo video.mp4 existe!\n");
            stop_signal = 1;
        }

        for (int i = 1; i < size; i++) MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        if (stop_signal) { MPI_Finalize(); return -1; }

        Mat frame;
        capture >> frame;
        if (frame.empty()) { MPI_Finalize(); return -1; }

        frame_info[0] = frame.cols;
        frame_info[1] = frame.rows;
        frame_info[2] = frame.channels();
        frame_info[3] = frame.step; 
        long data_size = frame.rows * frame.step;

        for (int i = 1; i < size; i++) MPI_Send(frame_info, 4, MPI_INT, i, 1, MPI_COMM_WORLD);

        std::vector<Mat> frames_received(size - 1);
        for (int i = 0; i < size - 1; i++) {
            frames_received[i] = Mat(frame.rows, frame.cols, frame.type());
        }

        int playing = 1, frame_id_counter = 0;

        printf("[Coordenador] Iniciando o processamento do video...\n");
        
        /* === INÍCIO DO CRONÔMETRO === */
        double start_time = MPI_Wtime();

        while (playing) {
            int workers_active = 0;

            /* 1. Despacha os frames */
            for (int i = 1; i < size; i++) {
                capture >> frame;
                if (frame.empty()) {
                    playing = 0;
                    break;
                }
                
                MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&frame_id_counter, 1, MPI_INT, i, 10, MPI_COMM_WORLD);
                MPI_Send(frame.data, data_size, MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD);
                
                frame_id_counter++;
                workers_active++;
            }

            /* 2. Recebe e exibe */
            for (int i = 1; i <= workers_active; i++) {
                int received_frame_id;
                MPI_Recv(&received_frame_id, 1, MPI_INT, i, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(frames_received[i-1].data, data_size, MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                imshow("Video Processado", frames_received[i-1]);
                
                if (waitKey(1) == 27) playing = 0; /* ESC para sair */
            }
        }

        /* === FIM DO CRONÔMETRO === */
        double end_time = MPI_Wtime();
        
        printf("\n==================================================\n");
        printf("[Coordenador] Fim do video!\n");
        printf("[Coordenador] Total de frames processados: %d\n", frame_id_counter);
        printf("[Coordenador] Tempo total de execucao: %.4f segundos\n", end_time - start_time);
        printf("==================================================\n\n");

        stop_signal = 1;
        for (int i = 1; i < size; i++) MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

        capture.release();
        destroyAllWindows();

    } else {
        /* ========================================== */
        /* PROCESSOS TRABALHADORES                    */
        /* ========================================== */
        MPI_Recv(&stop_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (stop_signal) { MPI_Finalize(); return 0; }

        MPI_Recv(frame_info, 4, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int width = frame_info[0], height = frame_info[1], channels = frame_info[2], widthStep = frame_info[3];
        long data_size = height * widthStep;

        unsigned char* input_buffer = new unsigned char[data_size];
        unsigned char* output_buffer = new unsigned char[data_size];

        while (1) {
            int current_frame_id;
            MPI_Recv(&stop_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (stop_signal) break;

            MPI_Recv(&current_frame_id, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(input_buffer, data_size, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            applyManualFilter(input_buffer, output_buffer, width, height, channels, widthStep, rank, current_frame_id);

            MPI_Send(&current_frame_id, 1, MPI_INT, 0, 11, MPI_COMM_WORLD);
            MPI_Send(output_buffer, data_size, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
        }

        delete[] input_buffer;
        delete[] output_buffer;
    }

    MPI_Finalize();
    return 0;
}
