#ifndef NNET_CONTROLLED_RELU_H_
#define NNET_CONTROLLED_RELU_H_
namespace nnet {
template<class data_T, class result_T, typename CONFIG_T>
void controlled_relu(data_T data[CONFIG_T::n_in], result_T result[CONFIG_T::n_in]) {
    #pragma HLS INLINE off
    relu_steps:
    for (unsigned step=0; step<CONFIG_T::steps; step++) {
        #pragma HLS PIPELINE II=1
        relu_lanes:
        for (unsigned lane=0; lane<CONFIG_T::lanes; lane++) {
            #pragma HLS UNROLL
            const unsigned i=step*CONFIG_T::lanes+lane;
            if (i<CONFIG_T::n_in) {
                data_T value=data[i];
                if (value>0) result[i]=value;
                else result[i]=0;
            }
        }
    }
}
}
#endif
