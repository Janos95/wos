#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
//
// Created by janos on 17.06.20.
//
#define CORRADE_STANDARD_ASSERT

#include "kdtree.h"
#include "arc_ball_camera.hpp"
#include "types.hpp"
#include "upload.hpp"

#include <scoped_timer/scoped_timer.hpp>

#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/StaticArray.h>

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/ImageView.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/DebugTools/ColorMap.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/Primitives/Plane.h>
#include <Magnum/Primitives/UVSphere.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Grid.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/DebugTools/ObjectRenderer.h>
#include <Magnum/Primitives/Axis.h>

#include <MagnumPlugins/AssimpImporter/AssimpImporter.h>

#include <random>

using namespace Magnum;
using namespace Corrade;

using namespace Magnum::Math::Literals;

//constexpr Vector3 lightPositions[6] = {
//        {10,0,0},
//        {-10,0,0},
//        {0,10,0},
//        {0,-10,0},
//        {0,0,10},
//        {0,0,-10}
//};
//
//constexpr Color4 lightColors[6] = {
//        Color4{1.f/3.f,1.f/3.f,1.f/3.f,1},
//        Color4{1.f/3.f,1.f/3.f,1.f/3.f,1},
//        Color4{1.f/3.f,1.f/3.f,1.f/3.f,1},
//        Color4{1.f/3.f,1.f/3.f,1.f/3.f,1},
//        Color4{1.f/3.f,1.f/3.f,1.f/3.f,1},
//        Color4{1.f/3.f,1.f/3.f,1.f/3.f,1},
//};

struct PhongDrawable : Drawable{
    explicit PhongDrawable(Object3D& obj, GL::Mesh& m, Shaders::Phong& s, DrawableGroup* group) :
            Drawable(obj, group),
            shader(s),
            mesh(m)
    {
    }

    void draw(const Matrix4& tf, SceneGraph::Camera3D& camera) override {
        if(texture)
            shader.bindDiffuseTexture(*texture);
        else
            shader.setDiffuseColor(color);

        shader.setShininess(200.0f)
              .setAmbientColor(0x111111_rgbf)
              .setSpecularColor(0x330000_rgbf)
              .setLightPosition({10.0f, 15.0f, 5.0f})
              .setTransformationMatrix(tf)
              .setNormalMatrix(tf.normalMatrix())
              .setProjectionMatrix(camera.projectionMatrix())
              .draw(mesh);
    }

    Shaders::Phong& shader;
    GL::Mesh& mesh;
    GL::Texture2D* texture = nullptr;
    Color4 color{0x808080_rgbf};
};

struct FlatDrawable : Drawable {
    explicit FlatDrawable(Object3D& obj, GL::Mesh& m, Shaders::Flat3D& s, DrawableGroup* group) :
            Drawable(obj, group),
            shader(s),
            mesh(m)
    {
    }

    void draw(const Matrix4& tf, SceneGraph::Camera3D& camera) override {

        shader.setTransformationProjectionMatrix(tf * camera.projectionMatrix())
              .setColor(color)
              .draw(mesh);
    }

    Shaders::Flat3D& shader;
    GL::Mesh& mesh;
    Color4 color{0x808080_rgbf};
};

struct VertexColorDrawable : Drawable{
    explicit VertexColorDrawable(Object3D& obj, GL::Mesh& m, Shaders::VertexColor3D& s, DrawableGroup* group) :
            Drawable(obj, group),
            shader(s),
            mesh(m)
    {
    }

    void draw(const Matrix4& tf, SceneGraph::Camera3D& camera) override {

        shader.setTransformationProjectionMatrix(tf * camera.projectionMatrix())
              .draw(mesh);
    }

    Shaders::VertexColor3D& shader;
    GL::Mesh& mesh;
};

class WoSApplication: public Platform::Application {
public:
    explicit WoSApplication(const Arguments& arguments);

private:
    void drawEvent() override;
    void viewportEvent(ViewportEvent& event) override;
    void keyPressEvent(KeyEvent& event) override;
    void mousePressEvent(MouseEvent& event) override;
    void mouseReleaseEvent(MouseEvent& event) override;
    void mouseMoveEvent(MouseMoveEvent& event) override;
    void mouseScrollEvent(MouseScrollEvent& event) override;

    void solve();

    Corrade::Containers::Optional<ArcBallCamera> camera;
    Scene3D scene;
    DrawableGroup drawables;
    Trade::MeshData meshData{MeshPrimitive::Points, 0};
    Trade::MeshData planeData{MeshPrimitive::Points, 0};
    GL::Mesh mesh{Magnum::NoCreate};
    GL::Mesh axis{Magnum::NoCreate};

    GL::Mesh plane{Magnum::NoCreate};
    GL::Buffer vertexBuffer{Mg::NoCreate}, indexBuffer{Mg::NoCreate};

    GL::Texture2D texture{Magnum::NoCreate};
    GL::Texture2D jet{Magnum::NoCreate};
    KDTree<Vector3> tree;
    Shaders::Phong phong{Magnum::NoCreate};
    Shaders::Phong phongTextured{Magnum::NoCreate};
    Shaders::Flat3D flat{Magnum::NoCreate};
    Shaders::VertexColor3D vertexColor{Magnum::NoCreate};
    PhongDrawable* meshDrawer{};
    PhongDrawable* planeDrawer{};

    Containers::Array<Float> boundaryValues;
    Containers::Array<float> solution;
    Containers::Array<float> delta;
    Containers::Array<int> samples;
    Containers::StridedArrayView1D<Vector2> textureView;
    Containers::StridedArrayView1D<const Vector3> gridView;
    uint32_t depth = 4;

    const float eps = 0.01; /* stopping tolerance */
    const float convEps = 0.01; /* if this is reached we refine the solution */
    const int nWalks = 32; /* number of Monte Carlo samples per iteration */
    const int maxSteps = 16; /* maximum walk length */
    const std::size_t gridSize = 512;

    std::size_t start = 0;
    std::size_t blockSize = 10;
    std::size_t minIdx = 0, maxIdx = 0;
    std::size_t subDiv = 0;
};

WoSApplication::WoSApplication(const Arguments& arguments):
    Platform::Application{arguments, Magnum::NoCreate}
{
/* Setup window */
    {
        const Vector2 dpiScaling = this->dpiScaling({});
        Configuration conf;
        conf.setTitle("Viewer")
                .setSize(conf.size(), dpiScaling)
                .setWindowFlags(Configuration::WindowFlag::Resizable);
        GLConfiguration glConf;
        glConf.setSampleCount(dpiScaling.max() < 2.0f ? 8 : 2);
        if(!tryCreate(conf, glConf)) {
            create(conf, glConf.setSampleCount(0));
        }
    }

    /* Set up the camera */
    {
        /* Setup the arcball after the camera objects */
        const Vector3 eye = Vector3::zAxis(-10.0f);
        const Vector3 center{};
        const Vector3 up = Vector3::yAxis();
        camera.emplace(scene, eye, center, up, 45.0_degf,
                               windowSize(), framebufferSize());
    }

    /* load assets and setup scene */
    {
        ScopedTimer timerScene("Setting up the whole scene", true);
        PluginManager::Manager<Trade::AbstractImporter> manager;

        auto sceneImporter = manager.loadAndInstantiate("AssimpImporter");
        if (!sceneImporter) std::exit(1);
        if (!sceneImporter->openFile("/home/janos/wos/assets/fj13.ply")) std::exit(2);

        if (sceneImporter->meshCount() && sceneImporter->mesh(0)) {
            meshData = *(sceneImporter->mesh(0));
            auto positions = meshData.mutableAttribute<Vector3>(Trade::MeshAttribute::Position);
            Vector3 average{0};
            for(auto const& p : positions)
                average += p;
            average /= float(meshData.vertexCount());
            Containers::arrayResize(boundaryValues, Containers::NoInit, meshData.vertexCount());
            for (uint32_t i = 0; i < meshData.vertexCount(); ++i) {
                positions[i] -= average;
                boundaryValues[i] = positions[i].dot();
            }

            mesh = MeshTools::compile(meshData, MeshTools::CompileFlag::GenerateSmoothNormals);
            ScopedTimer timer{"Building Kd Tree", true};
            tree = KDTree{meshData.attribute<Vector3>(Trade::MeshAttribute::Position)};
        } else std::exit(3);

        phong = Shaders::Phong{};
        phongTextured = Shaders::Phong{Shaders::Phong::Flag::DiffuseTexture};
        vertexColor = Shaders::VertexColor3D{};

        auto object = new Object3D{&scene};
        meshDrawer = new PhongDrawable{*object, mesh, phong, &drawables};
        axis = MeshTools::compile(Primitives::axis3D());
        new VertexColorDrawable{*object, axis, vertexColor, &drawables};

        const auto map = DebugTools::ColorMap::turbo();
        const Vector2i size{Int(map.size()), 1};

        jet = GL::Texture2D{};
        jet.setMinificationFilter(SamplerFilter::Linear)
           .setMagnificationFilter(SamplerFilter::Linear)
           .setWrapping(SamplerWrapping::ClampToEdge) // or Repeat
           .setStorage(1, GL::TextureFormat::RGB8, size) // or SRGB8
           .setSubImage(0, {}, ImageView2D{PixelFormat::RGB8Srgb, size, map});

        planeData = Primitives::grid3DSolid({(int)gridSize, (int)gridSize}, Primitives::GridFlag::TextureCoordinates|Primitives::GridFlag::Normals);
        gridView = planeData.attribute<Vector3>(Trade::MeshAttribute::Position);
        Containers::arrayResize(solution, gridView.size());
        Containers::arrayResize(samples, gridView.size());
        Containers::arrayResize(delta, gridView.size());

        textureView = planeData.mutableAttribute<Vector2>(Trade::MeshAttribute::TextureCoordinates);
        for(auto& p : textureView)
            p = {0,0};

        plane = Magnum::GL::Mesh{};
        vertexBuffer = Magnum::GL::Buffer{};
        indexBuffer = Magnum::GL::Buffer{};
        Debug{} << "uploading plane to the gpu";
        upload(plane, vertexBuffer, indexBuffer, planeData);
        Debug{} << "done uploading";
        planeDrawer = new PhongDrawable{*object, plane, phongTextured, &drawables};
        planeDrawer->texture = &jet;

        Debug{} << "done setting up the scene";
    }

    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

    /* Start the timer, loop at 60 Hz max */
    setSwapInterval(1);
    setMinimalLoopPeriod(16);
}

void WoSApplication::solve(){
    auto end = Math::min(solution.size(), start + blockSize);
    std::fill(delta.begin() + start, delta.begin() + end, 100.f);
    std::random_device dev;
    auto engine = std::default_random_engine{dev()};
    auto distr = std::normal_distribution<float>{};
#pragma omp parallel for
    for (uint32_t j = start; j < end; ++j) {
            float sum = 0.;
            int nRealizedWalks = 0;
            for(int i = 0; i < nWalks; ++i) {
                Vector3 x = gridView[j];
                float dist;
                int steps = 0;
                int idx;
                do {
                    auto result = tree.nearestNeighbor(x);
                    dist = sqrt(result.distanceSquared);
                    idx = result.pointIndex;
                    x = x + dist * Vector3(distr(engine), distr(engine), distr(engine)).normalized();
                    steps++;
                }
                while( dist > eps && steps < maxSteps );

                if(dist <= eps){
                    ++nRealizedWalks;
                    sum += boundaryValues[idx];
                }
            }
            auto& value = solution[j];
            auto& numSamples = samples[j];
            if(nRealizedWalks) {
                float newValue = (float(numSamples) * value + sum) / float(nRealizedWalks + numSamples);
                delta[j] = abs(newValue - value);
                value = newValue;
                numSamples += nWalks;
            }
    };

    auto [min, max] = Math::minmax(solution);
    for (uint32_t j = start; j < end; ++j) {
        textureView[j].x() = (solution[j] - min) / (max - min);
    }
    reuploadVertices(vertexBuffer, planeData);

    start = end % solution.size();
    auto minDelta = *std::min_element(delta.begin(), delta.end(), )
}

void WoSApplication::drawEvent() {

    /* step the monte carlo solver */
    solve();

    GL::defaultFramebuffer.clear(
            GL::FramebufferClear::Color|GL::FramebufferClear::Depth);
    camera->update();
    camera->draw(drawables);

    swapBuffers();

    /* run next frame immediately */
    redraw();
}

void WoSApplication::viewportEvent(ViewportEvent& event) {
    GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});
    camera->reshape(event.windowSize(), event.framebufferSize());
}

void WoSApplication::keyPressEvent(KeyEvent& event) {
    switch(event.key()) {
        case KeyEvent::Key::L:
            if(camera->lagging() > 0.0f) {
                Debug{} << "Lagging disabled";
                camera->setLagging(0.0f);
            } else {
                Debug{} << "Lagging enabled";
                camera->setLagging(0.85f);
            }
            break;
        case KeyEvent::Key::R:
            camera->reset();
            break;
        default: return;
    }

    event.setAccepted();
    redraw(); /* camera has changed, redraw! */
}

void WoSApplication::mousePressEvent(MouseEvent& event) {
    /* Enable mouse capture so the mouse can drag outside of the window */
    /** @todo replace once https://github.com/mosra/magnum/pull/419 is in */
    SDL_CaptureMouse(SDL_TRUE);

    camera->initTransformation(event.position());

    event.setAccepted();
    redraw(); /* camera has changed, redraw! */
}

void WoSApplication::mouseReleaseEvent(MouseEvent&) {
    /* Disable mouse capture again */
    /** @todo replace once https://github.com/mosra/magnum/pull/419 is in */
    SDL_CaptureMouse(SDL_FALSE);
}

void WoSApplication::mouseMoveEvent(MouseMoveEvent& event) {
    if(!event.buttons()) return;

    if(event.modifiers() & MouseMoveEvent::Modifier::Shift)
        camera->translate(event.position());
    else camera->rotate(event.position());

    event.setAccepted();
    redraw(); /* camera has changed, redraw! */
}

void WoSApplication::mouseScrollEvent(MouseScrollEvent& event) {
    const Float offsetDelta = event.offset().y();
    if(Math::abs(offsetDelta) < 1.0e-2f) return;

    camera->zoom(offsetDelta);

    event.setAccepted();
    redraw(); /* camera has changed, redraw! */
}

MAGNUM_APPLICATION_MAIN(WoSApplication)

#pragma clang diagnostic pop