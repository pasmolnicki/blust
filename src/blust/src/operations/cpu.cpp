#include <blust/backend/cpu_ops.hpp>

START_BLUST_NAMESPACE

ops_tensor cpu_ops::add(ops_tensor a, ops_tensor b)
{
    M_assert_tensor_same_size(a, b);
    ops_tensor res(new number_t[a.size()], a.layout());

    // Perform a tiled addition of the 2 tensors
    auto a_data = a.data();
    auto b_data = b.data();
    auto c_data = res.data();
    int i = 0, size = res.size();

    register double res_a0, res_a1, res_a2, res_a3;

    res_a0 = 0.0;
    res_a1 = 0.0;
    res_a2 = 0.0;
    res_a3 = 0.0;

    for (; i < size; i += 4)
    {
        res_a0 = *a_data;
        res_a1 = *(a_data + 1);
        res_a2 = *(a_data + 2);
        res_a3 = *(a_data + 3);

        res_a0 += *b_data;
        res_a1 += *(b_data + 1);
        res_a2 += *(b_data + 2);
        res_a3 += *(b_data + 3);

        *c_data       = res_a0;
        *(c_data + 1) = res_a1;
        *(c_data + 2) = res_a2;
        *(c_data + 3) = res_a3;
        
        a_data += 4;
        b_data += 4;
        c_data += 4;
    }

    // Add the rest of the elements
    for (;i < size; i++, a_data++, b_data++, c_data++) {
        *c_data += *a_data + *b_data;
    }

    return res;
}


END_BLUST_NAMESPACE