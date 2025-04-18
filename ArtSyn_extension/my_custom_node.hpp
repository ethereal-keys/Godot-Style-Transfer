#ifndef MY_CUSTOM_NODE_HPP
#define MY_CUSTOM_NODE_HPP

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/sub_viewport.hpp>
#include <godot_cpp/classes/viewport_texture.hpp>
#include <godot_cpp/classes/sprite2d.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/image_texture.hpp>  
#include <torch/script.h>  

namespace godot {
    class MyCustomNode : public Node {
        GDCLASS(MyCustomNode, Node);

    private:
        SubViewport* viewport;
        Sprite2D* display_sprite;
        torch::jit::script::Module model;  // Style transfer model
        Ref<ImageTexture> display_texture;  // Persistent texture
        Ref<Image> process_image(const Ref<Image>& input_image);

    protected:
        static void _bind_methods();

    public:
        MyCustomNode();
        ~MyCustomNode();
        void _ready() override;
        void _process(double delta) override;
        void set_viewport(NodePath path);
        NodePath get_viewport() const;
        void set_display_sprite(NodePath path);
        NodePath get_display_sprite() const;
        String say_hello() const;
    };
}
#endif
