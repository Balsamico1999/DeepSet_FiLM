#ifndef NNET_INDEXED_BROADCAST_H_
#define NNET_INDEXED_BROADCAST_H_

namespace nnet {
// Flat row-major arrays, with zero strides on singleton input dimensions.
// Reuse controls arithmetic lanes, rather than constructing replicated operands.
template<class a_T, class b_T, class result_T, typename CONFIG_T>
void indexed_broadcast(a_T a[CONFIG_T::ar * CONFIG_T::ac],
                       b_T b[CONFIG_T::br * CONFIG_T::bc],
                       result_T result[CONFIG_T::rows * CONFIG_T::cols]) {
    #pragma HLS INLINE off
    broadcast_steps:
    for (unsigned step = 0; step < CONFIG_T::steps; step++) {
        #pragma HLS PIPELINE II=1
        broadcast_lanes:
        for (unsigned lane = 0; lane < CONFIG_T::lanes; lane++) {
            #pragma HLS UNROLL
            const unsigned i = step * CONFIG_T::lanes + lane;
            if (i < CONFIG_T::rows * CONFIG_T::cols) {
                const unsigned row = i / CONFIG_T::cols;
                const unsigned col = i % CONFIG_T::cols;
                const unsigned ai = (CONFIG_T::ar == 1 ? 0 : row) * CONFIG_T::ac
                                  + (CONFIG_T::ac == 1 ? 0 : col);
                const unsigned bi = (CONFIG_T::br == 1 ? 0 : row) * CONFIG_T::bc
                                  + (CONFIG_T::bc == 1 ? 0 : col);
                if (CONFIG_T::multiply)
                    result[i] = a[ai] * b[bi];
                else
                    result[i] = a[ai] + b[bi];
            }
        }
    }
}
}
#endif
