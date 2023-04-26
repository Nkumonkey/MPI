#include<iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include<Windows.h>
using namespace std;
const int N = 1000;
void LU(float** A, int rank, int num_proc)
 {
		 int block = N / num_proc, remain = N % num_proc;
 int begin = rank * block;
	 int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
		 for (int k = 0; k < N; k++) {
		 //当 前 行 是 自 己 进 程 的 任 务 — — 进 行 消 去
			 if(int(k % num_proc) == rank) {
			 for (int j = k + 1; j < N; j++)
				 A[k][j] = A[k][j] / A[k][k];
			 A[k][k] = 1.0;
			 //发 送 消 息 （向 所 有 其 他 进 程）
				 for (int p = 0; p < num_proc; p++)
				 if(p != rank)
				 MPI_Send(A[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
		}
		 //当 前 行 不 是 自 己 进 程 的 任 务 — — 接 收 消 息
			 else {
				  MPI_Recv(A[k], N, MPI_FLOAT, int(k % num_proc), 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			 }

		 //消 去 部 分
			
			 for (int i = k + 1; i < N; i++)
				 if (int(i % num_proc) == rank) {
					 for (int j = k + 1; j < N; j++)
						 A[i][j] = A[i][j] - A[i][k] * A[k][j];
					 A[i][k] = 0.0;
				 }
}}
void f_mpi()
 {
	float** A;
	A = new float* [N];
	for (int i = 0; i < N; ++i)
		A[i] = new float [N];
	MPI_Init(NULL,NULL);
	 int num_proc; //进 程 数
		 int rank; //识 别 调 用 进 程 的 rank ， 值 从0~ size −1
		 MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
		 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		int block = N / num_proc, remain = N % num_proc;
	 //0号 进 程 — — 任 务 划 分
		if(rank == 0) {
			srand(time(NULL));
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N; j++)
				{
					A[i][j] = rand(); // 生成一个 0 到 9 的随机数
				}
			}
			long long head, tail, freq;
			QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
			QueryPerformanceCounter((LARGE_INTEGER*)&head);
		//任 务 划 分
			for (int i = 0; i < N; i++) {
				//根 据 行 号 除 进 程 数 的 余 数 进 行 划 分
				int flag = i % num_proc;
				if (flag == rank)
					continue;
				else
					MPI_Send(A[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
			}
			LU(A, rank, num_proc);
		//处 理 完0号 进 程 自 己 的 任 务 后 需 接 收 其 他 进 程 处 理 之 后 的 结 果
			for (int i = 0; i < N; i++) {
				//根 据 行 号 除 进 程 数 的 余 数 接 收 结 果
				int flag = i % num_proc;
				if (flag == rank)
					continue;
				else
					MPI_Recv(A[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			QueryPerformanceCounter((LARGE_INTEGER *) & tail);
		cout << "Block MPI LU time cost: "<< (tail - head) *1000.0 / freq << "ms" << endl;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j)
				cout << A[i][j] << " ";
			cout << endl;
		}
		MPI_Finalize();
	}
		//其 他 进 程
		else {
			//非0号 进 程 先 接 收 任 务
			for (int i = rank; i < N; i += num_proc) {
				//每 间 隔num_proc行 接 收 一 行 数 据
				MPI_Recv(A[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			LU(A, rank, num_proc);
			//非0号 进 程 完 成 任 务 之 后， 将 结 果 传 回 到0号 进 程
			for (int i = rank; i < N; i += num_proc) {
				//每 间 隔num_proc行 发 送 一 行 数 据
				MPI_Send(A[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
			}
			MPI_Finalize();
	}
}
int main() {
	f_mpi();
	return 0;
} 
