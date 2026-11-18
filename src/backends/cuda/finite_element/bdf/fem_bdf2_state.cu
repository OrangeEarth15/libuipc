#include <finite_element/bdf/fem_bdf2_state.h>
#include <finite_element/finite_element_method.h>

namespace uipc::backend::cuda
{
namespace
{
    __global__ void fem_bdf2_state_restart_history_kernel(
        cuda_tool::CBufferView<IndexT>  is_fixed,
        cuda_tool::CBufferView<IndexT>  is_dynamic,
        cuda_tool::CBufferView<Vector3> xs,
        cuda_tool::CBufferView<Vector3> vs,
        cuda_tool::BufferView<Vector3>  x_n_1s,
        cuda_tool::BufferView<Vector3>  v_n_1s,
        Float                           dt,
        int                             n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        // A fixed or quasi-static vertex has no history to extrapolate: it
        // starts at rest where it is, as the prediction treats it.
        if(is_fixed(i) || !is_dynamic(i))
        {
            x_n_1s(i) = xs(i);
            v_n_1s(i).setZero();
            return;
        }
        x_n_1s(i) = xs(i) - dt * vs(i);
        v_n_1s(i) = vs(i);
    }
}  // namespace

REGISTER_SIM_SYSTEM(FEMBDF2State);

void FEMBDF2State::do_build()
{
    m_finite_element_method = require<FiniteElementMethod>();
    m_dt_attr               = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_dt_attr, "Scene config must have a 'dt' attribute.");
}

void FEMBDF2State::resize(SizeT size)
{
    m_impl.x_n_1s.resize(size);
    m_impl.v_n_1s.resize(size);
}

void FEMBDF2State::restart_history(SizeT offset, SizeT count)
{
    // Before the first step the history is empty: the integrator starts it
    // for every row then.
    int n = (int)count;
    if(n == 0 || m_impl.x_n_1s.size() == 0)
        return;
    auto xs       = m_finite_element_method->xs().subview(offset, count);
    auto vs       = m_finite_element_method->vs().subview(offset, count);
    auto is_fixed = m_finite_element_method->is_fixed().subview(offset, count);
    auto is_dynamic = m_finite_element_method->is_dynamic().subview(offset, count);
    Float dt = m_dt_attr->view()[0];
    auto  k  = fem_bdf2_state_restart_history_kernel;
    k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
        is_fixed,
        is_dynamic,
        xs,
        vs,
        m_impl.x_n_1s.view(offset, count),
        m_impl.v_n_1s.view(offset, count),
        dt,
        n);
}

bool FEMBDF2State::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_x_n_1s.dump(fmt::format("{}q_n_1.{}", path, frame), x_n_1s)  //
           && dump_v_n_1s.dump(fmt::format("{}qv_n_1.{}", path, frame), v_n_1s);
}

bool FEMBDF2State::Impl::try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_x_n_1s.load(fmt::format("{}q_n_1.{}", path, frame))  //
           && dump_v_n_1s.load(fmt::format("{}qv_n_1.{}", path, frame));
}

void FEMBDF2State::Impl::apply_recover(RecoverInfo& info)
{
    dump_x_n_1s.apply_to(x_n_1s);
    dump_v_n_1s.apply_to(v_n_1s);
}

void FEMBDF2State::Impl::clear_recover(RecoverInfo& info)
{
    dump_x_n_1s.clean_up();
    dump_v_n_1s.clean_up();
}

bool FEMBDF2State::do_dump(DumpInfo& info)
{
    return m_impl.dump(info);
}

bool FEMBDF2State::do_try_recover(RecoverInfo& info)
{
    return m_impl.try_recover(info);
}

void FEMBDF2State::do_apply_recover(RecoverInfo& info)
{
    m_impl.apply_recover(info);
}

void FEMBDF2State::do_clear_recover(RecoverInfo& info)
{
    m_impl.clear_recover(info);
}
}  // namespace uipc::backend::cuda
