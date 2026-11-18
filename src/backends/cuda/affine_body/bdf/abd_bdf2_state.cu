#include <affine_body/bdf/abd_bdf2_state.h>
#include <affine_body/affine_body_dynamics.h>

namespace uipc::backend::cuda
{
namespace
{
    __global__ void abd_bdf2_state_restart_history_kernel(
        cuda_tool::CBufferView<IndexT>   is_fixed,
        cuda_tool::CBufferView<IndexT>   is_dynamic,
        cuda_tool::CBufferView<Vector12> qs,
        cuda_tool::CBufferView<Vector12> q_vs,
        cuda_tool::BufferView<Vector12>  q_n_1s,
        cuda_tool::BufferView<Vector12>  q_v_n_1s,
        Float                            dt,
        int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        // A fixed or quasi-static body has no history to extrapolate: it
        // starts at rest where it is, as the prediction treats it.
        if(is_fixed(i) || !is_dynamic(i))
        {
            q_n_1s(i) = qs(i);
            q_v_n_1s(i).setZero();
            return;
        }
        q_n_1s(i)   = qs(i) - dt * q_vs(i);
        q_v_n_1s(i) = q_vs(i);
    }
}  // namespace

REGISTER_SIM_SYSTEM(ABDBDF2State);

void ABDBDF2State::do_build()
{
    m_affine_body_dynamics = require<AffineBodyDynamics>();
    m_dt_attr              = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_dt_attr, "Scene config must have a 'dt' attribute.");
}

void ABDBDF2State::resize(SizeT size)
{
    m_impl.q_n_1s.resize(size);
    m_impl.q_v_n_1s.resize(size);
}

void ABDBDF2State::restart_history(SizeT offset, SizeT count)
{
    // Before the first step the history is empty: the integrator starts it
    // for every row then.
    int n = (int)count;
    if(n == 0 || m_impl.q_n_1s.size() == 0)
        return;
    auto qs   = m_affine_body_dynamics->qs().subview(offset, count);
    auto q_vs = m_affine_body_dynamics->q_vs().subview(offset, count);
    auto is_fixed = m_affine_body_dynamics->body_is_fixed().subview(offset, count);
    auto is_dynamic = m_affine_body_dynamics->body_is_dynamic().subview(offset, count);
    Float dt = m_dt_attr->view()[0];
    auto  k  = abd_bdf2_state_restart_history_kernel;
    k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
        is_fixed,
        is_dynamic,
        qs,
        q_vs,
        m_impl.q_n_1s.view(offset, count),
        m_impl.q_v_n_1s.view(offset, count),
        dt,
        n);
}

bool ABDBDF2State::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_q_n_1s.dump(fmt::format("{}q_n_1.{}", path, frame), q_n_1s)  //
           && dump_q_v_n_1s.dump(fmt::format("{}qv_n_1.{}", path, frame), q_v_n_1s);
}

bool ABDBDF2State::Impl::try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_q_n_1s.load(fmt::format("{}q_n_1.{}", path, frame))  //
           && dump_q_v_n_1s.load(fmt::format("{}qv_n_1.{}", path, frame));
}

void ABDBDF2State::Impl::apply_recover(RecoverInfo& info)
{
    dump_q_n_1s.apply_to(q_n_1s);
    dump_q_v_n_1s.apply_to(q_v_n_1s);
}

void ABDBDF2State::Impl::clear_recover(RecoverInfo& info)
{
    dump_q_n_1s.clean_up();
    dump_q_v_n_1s.clean_up();
}

bool ABDBDF2State::do_dump(DumpInfo& info)
{
    return m_impl.dump(info);
}

bool ABDBDF2State::do_try_recover(RecoverInfo& info)
{
    return m_impl.try_recover(info);
}

void ABDBDF2State::do_apply_recover(RecoverInfo& info)
{
    m_impl.apply_recover(info);
}

void ABDBDF2State::do_clear_recover(RecoverInfo& info)
{
    m_impl.clear_recover(info);
}
}  // namespace uipc::backend::cuda
