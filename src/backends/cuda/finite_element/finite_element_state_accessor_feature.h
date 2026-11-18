#pragma once
#include <type_define.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <cuda_tool/cuda_tool.h>
#include <sim_system.h>

namespace uipc::backend::cuda
{
class FiniteElementMethod;
class FiniteElementVertexReporter;
class FEMBDF2State;

class FiniteElementStateAccessorFeatureOverrider final : public core::FiniteElementStateAccessorFeatureOverrider
{
  public:
    FiniteElementStateAccessorFeatureOverrider(FiniteElementMethod& fem,
                                               FiniteElementVertexReporter& vertex_reporter,
                                               FEMBDF2State* bdf2_state);

    SizeT get_vertex_count() override;
    void  do_copy_from(const geometry::SimplicialComplex& state_geo) override;
    void  do_copy_to(geometry::SimplicialComplex& state_geo) override;

    void do_copy_position_to(backend::BufferView buffer_view,
                             IndexT              vertex_offset,
                             SizeT               vertex_count) override;
    void do_copy_velocity_to(backend::BufferView buffer_view,
                             IndexT              vertex_offset,
                             SizeT               vertex_count) override;

  private:
    FiniteElementMethod&         m_fem;
    FiniteElementVertexReporter& m_vertex_reporter;
    // the BDF2 history (x^{n-1}, \dot{x}^{n-1}), restarted for written vertices
    SimSystemSlot<FEMBDF2State> m_bdf2_state;
};
}  // namespace uipc::backend::cuda
