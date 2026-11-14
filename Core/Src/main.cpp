#include "main.hpp"

using namespace std;

int main() {
  vector<size_t> dim = {2, 3, 4};
  Tensor<float> aTensor(dim);
  cout << aTensor;
  return 0;
}