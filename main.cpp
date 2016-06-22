#include "libs/CImg.h"
#include <iostream>
#include "meanshift.hpp"

#define cimg_use_magick

using namespace cimg_library;

// g++ main.cpp -o main.out -L/opt/X11/lib -lX11 -pthread -std=c++11 -I/opt/ann/include -L/opt/ann/lib -lANN

// carregar todos os pontos iniciais para um kdtree (ANN)
// Duplicar todos os pontos iniciais, criando os pontos moveis
// procurar por 50 vizinhos mais proximos. Andar com o ponto movel

// achou o ponto final, depois de parar de convergir. Para achar a cor deste ponto, 
// olhar na imagem inicial a partir do ponto xy

int main(int argc, char * argv[]) 
{
	CImg<double> image("images/input/marroquim.png");

	Meanshift meanshift;
	meanshift.filtering(image, 0.0, 0.0);
	return 0;
}