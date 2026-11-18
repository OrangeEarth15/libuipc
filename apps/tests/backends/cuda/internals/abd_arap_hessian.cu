#include <app/app.h>
#include <affine_body/constitutions/arap_function.h>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <random>

namespace
{
using namespace uipc;
using namespace uipc::backend::cuda;
namespace ARAP = sym::abd_arap;

Matrix3x3 random_rotation(std::mt19937_64& gen)
{
    std::normal_distribution<Float> n;
    Matrix3x3 M = Matrix3x3::NullaryExpr([&] { return n(gen); });
    Matrix3x3 Q = Eigen::HouseholderQR<Matrix3x3>(M).householderQ();
    if(Q.determinant() < 0)
        Q.col(0) *= -1;
    return Q;
}

// The gradient of |F - R|_F^2, 2 (F - R) with R the rotation of the polar
// decomposition, laid out like q: entry 3 r + c is entry (r, c).
Vector9 gradient(const Vector9& rows)
{
    Matrix3x3 F;
    for(int r = 0; r < 3; ++r)
        F.row(r) = rows.segment<3>(3 * r).transpose();
    Eigen::JacobiSVD<Matrix3x3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix3x3 G = 2 * (F - svd.matrixU() * svd.matrixV().transpose());
    Vector9   g;
    for(int r = 0; r < 3; ++r)
        g.segment<3>(3 * r) = G.row(r).transpose();
    return g;
}

// Central differences of the gradient with respect to the rows of F, symmetrized.
Matrix9x9 differences_of_the_gradient(const Matrix3x3& F)
{
    Vector9 rows;
    for(int r = 0; r < 3; ++r)
        rows.segment<3>(3 * r) = F.row(r).transpose();
    const Float h = 1e-6;
    Matrix9x9   H;
    for(int k = 0; k < 9; ++k)
    {
        Vector9 xp = rows, xm = rows;
        xp(k) += h;
        xm(k) -= h;
        H.col(k) = (gradient(xp) - gradient(xm)) / (2 * h);
    }
    return (0.5 * (H + H.transpose())).eval();
}

Matrix9x9 clamp_negative_eigenvalues(const Matrix9x9& H)
{
    Eigen::SelfAdjointEigenSolver<Matrix9x9> eig(H);
    Vector9 l = eig.eigenvalues().cwiseMax(0.0);
    return eig.eigenvectors() * l.asDiagonal() * eig.eigenvectors().transpose();
}
}  // namespace

TEST_CASE("abd_arap_hessian", "[cuda][abd][arap]")
{
    // The ARAP Hessian (constitution UID 2) with respect to the rows of F, as
    // q holds them, is the derivative of the gradient 2 (F - R) projected to
    // positive semi-definite: equal to the differences of the gradient where
    // the exact Hessian is positive definite (a stretched body), equal to
    // their eigenvalue clamp where it is not (a compressed one,
    // s_a + s_b < 2). F = R1 diag(s) R2 with random rotations.
    std::mt19937_64 gen(2);
    std::uniform_real_distribution<Float> stretched(1.05, 1.3), compressed(0.7, 0.95);

    struct Case
    {
        const char* name;
        Vector3     s;
    };
    const Case cases[] = {
        {"stretched", Vector3{stretched(gen), stretched(gen), stretched(gen)}},
        {"compressed", Vector3{compressed(gen), compressed(gen), compressed(gen)}},
        {"mixed", Vector3{0.8, 1.0, 1.3}},
    };

    for(const auto& c : cases)
    {
        INFO(c.name);
        Matrix3x3 F = random_rotation(gen) * c.s.asDiagonal() * random_rotation(gen);

        Matrix9x9 H;
        ARAP::ARAP_Hessian(H, F);
        Matrix9x9 exact = differences_of_the_gradient(F);
        Float     scale = exact.cwiseAbs().maxCoeff();

        // the matrix the solver gets: the projection of the exact Hessian
        Matrix9x9 projected = clamp_negative_eigenvalues(exact);
        CHECK((H - projected).cwiseAbs().maxCoeff() <= 1e-6 * scale);
        CHECK((H - H.transpose()).cwiseAbs().maxCoeff() <= 1e-12 * scale);
        CHECK(Eigen::SelfAdjointEigenSolver<Matrix9x9>(H).eigenvalues().minCoeff()
              >= -1e-9 * scale);

        Float lowest =
            Eigen::SelfAdjointEigenSolver<Matrix9x9>(exact).eigenvalues().minCoeff();
        if(std::string{c.name} == "stretched")
        {
            // nothing to project: H is the exact Hessian
            CHECK(lowest > 0);
            CHECK((H - exact).cwiseAbs().maxCoeff() <= 1e-6 * scale);
        }
        else
        {
            CHECK(lowest < -1e-3 * scale);  // the case exercises the projection
        }

        // The comparison tells the layouts apart: the same Hessian with F
        // flattened column by column, placed on the rows of F, is far off.
        Matrix9x9 P = Matrix9x9::Zero();
        for(int r = 0; r < 3; ++r)
            for(int col = 0; col < 3; ++col)
                P(3 * r + col, r + 3 * col) = 1;
        CHECK((P.transpose() * exact * P - exact).cwiseAbs().maxCoeff() > 0.05 * scale);
    }
}
