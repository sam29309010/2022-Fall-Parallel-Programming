__kernel void convolution(
    const int half_filter,
    const int imageWidth,
    const int imageHeight,
    __constant const float *d_filter,
    __global const float *d_in_img,
    __global float4* d_out_img
)
{
    // Should assure the imageWidth is divisible by 4
    const int global_id = get_global_id(0) << 2;
    const int center_width = global_id % imageWidth;
    const int center_height = global_id / imageWidth;
    const int filter_size = half_filter << 1 + 1;

    int filter_idx = 0;
    float4 sums = 0;
    for (int ker_height_offset = -half_filter; ker_height_offset <= half_filter; ker_height_offset++)
    {
        const int height = center_height + ker_height_offset;
        if ((height < 0) || (height >= imageHeight))
        {
            filter_idx += filter_size;
            continue;
        }
        
        const int height_idx = height * imageWidth;
        for (int ker_width_offset = -half_filter; ker_width_offset <= half_filter; ker_width_offset++, filter_idx++)
        {
            if (d_filter[filter_idx] != 0)
            {
                const int width = center_width + ker_width_offset;
                if ((width < 0) || (width >= imageWidth))
                    continue;

                const int idx = height_idx + width;
                const float4 inputs = {d_in_img[idx+0], d_in_img[idx+1], d_in_img[idx+2], d_in_img[idx+3]};
                const float filters = d_filter[filter_idx];
                sums += inputs * filters;
            }
        }
    }
    d_out_img[global_id >> 2] = sums;
}
