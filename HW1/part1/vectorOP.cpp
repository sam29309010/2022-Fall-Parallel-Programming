#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  __pp_vec_float x, result; // TBD
  __pp_vec_float max_value = _pp_vset_float(9.999999f);
  __pp_vec_int y = _pp_vset_int(0);
  __pp_vec_int ones = _pp_vset_int(1);
  __pp_vec_int zeros = _pp_vset_int(0);
  // valid, zero-exponent, positive-exponent, clamped masking 
  __pp_mask maskAll, maskZeroExp, maskPosExp, maskClamp;
  for (int i = 0; i < N; i += VECTOR_WIDTH){

    // initialization
    maskAll = _pp_init_ones(N-i);
    maskZeroExp = _pp_init_ones();
    maskClamp = _pp_init_ones(0);
    _pp_vload_float(x, values+i, maskAll);
    _pp_vload_int(y, exponents+i, maskAll);

    // zero-exponent
    _pp_vset_float(result, 1.f, maskAll);
    
    // one-exponent
    _pp_veq_int(maskZeroExp, y, zeros, maskAll);
    maskPosExp = _pp_mask_not(maskZeroExp);
    _pp_vmove_float(result, x, maskPosExp);
    _pp_vsub_int(y, y, ones, maskPosExp);

    // remaining-exponent
    // padding index should be invalidated (mask = 0)
    _pp_vgt_int(maskPosExp, y, zeros, maskAll);               // positive exponents
    _pp_vlt_float(maskPosExp, result, max_value, maskPosExp); // overflow
    while (_pp_cntbits(maskPosExp)){
      _pp_vmult_float(result, result, x, maskPosExp);
      _pp_vsub_int(y, y, ones, maskPosExp);
      _pp_vgt_int(maskPosExp, y, zeros, maskAll);
      _pp_vlt_float(maskPosExp, result, max_value, maskPosExp);
    }
    // clamping
    _pp_vgt_float(maskClamp, result, max_value, maskAll);
    _pp_vmove_float(result, max_value, maskClamp);

    // write back
    _pp_vstore_float(output+i, result, maskAll);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  float results[VECTOR_WIDTH];
  int vec_width = VECTOR_WIDTH >> 1;
  __pp_vec_float x;
  __pp_vec_float sum = _pp_vset_float(0.f);
  __pp_mask maskAll;

  // inter-vector summation
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones(N-i);
    _pp_vload_float(x, values + i, maskAll);
    _pp_vadd_float(sum, sum, x, maskAll);
  }

  // intra-vector summation
  // assuming the VECTOR_WIDTH is an exponent of 2
  while (vec_width){
    _pp_hadd_float(sum, sum);
    _pp_interleave_float(sum, sum);
    vec_width >>= 1;
  }

  // write back
  _pp_vstore_float(results, sum, maskAll);
  return results[0];
}