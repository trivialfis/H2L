import gi
import cv2
import sys

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, Gio
from gi.repository import GdkPixbuf


# MENU_XML = """
# <?xml version="1.0" encoding="UTF-8"?>
# <interface>
#   <menu id="app-menu">
#     <section>
#       <attribute name="label" translatable="yes">Change label</attribute>
#       <item>
#         <attribute name="action">win.change_label</attribute>
#         <attribute name="target">String 1</attribute>
#         <attribute name="label" translatable="yes">String 1</attribute>
#       </item>
#       <item>
#         <attribute name="action">win.change_label</attribute>
#         <attribute name="target">String 2</attribute>
#         <attribute name="label" translatable="yes">String 2</attribute>
#       </item>
#       <item>
#         <attribute name="action">win.change_label</attribute>
#         <attribute name="target">String 3</attribute>
#         <attribute name="label" translatable="yes">String 3</attribute>
#       </item>
#     </section>
#     <section>
#       <item>
#         <attribute name="action">win.maximize</attribute>
#         <attribute name="label" translatable="yes">Maximize</attribute>
#       </item>
#     </section>
#     <section>
#       <item>
#         <attribute name="action">app.about</attribute>
#         <attribute name="label" translatable="yes">_About</attribute>
#       </item>
#       <item>
#         <attribute name="action">app.quit</attribute>
#         <attribute name="label" translatable="yes">_Quit</attribute>
#         <attribute name="accel">&lt;Primary&gt;q</attribute>
#     </item>
#     </section>
#   </menu>
# </interface>
# """


def main():
    win = Gtk.Window(title='H2L')
    win.connect('delete-event', Gtk.main_quit)
    win.show_all()
    Gtk.main()


class H2L_WINDOW(Gtk.ApplicationWindow):

    def __init__(self, *args, **kwargs):
        # Gtk.Window.__init__(self, title='H2L')
        super().__init__(*args, **kwargs)

        self.vbox = Gtk.VBox(spacing=6)

        button = Gtk.Button('Choose File')
        button.connect('clicked', self.on_file_clicked)
        # self.text_view = Gtk.TextView()
        # self.text_buffer = self.text_view.get_buffer()
        self.vbox.pack_start(button, True, True, 0)
        # self.vbox.add(self.text_view)

        self.add(self.vbox)
        self.set_default_size(800, 600)

    def on_file_clicked(self, widget):
        dialog = Gtk.FileChooserDialog('Please choose a image', self,
                                       Gtk.FileChooserAction.OPEN,
                                       (Gtk.STOCK_CANCEL,
                                        Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN,
                                        Gtk.ResponseType.OK))

        self.add_filters(dialog)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            print('Open clicked')
            filename = dialog.get_filename()
            print('filename: ', filename)
        elif response == Gtk.ResponseType.CANCEL:
            filename = None
            print('Cancel clicked')
        dialog.destroy()

        if filename is not None:
            pixbuf = GdkPixbuf.PixbufAnimation.new_from_file(filename)
            pixbuf = pixbuf.get_static_image()

            w, h = pixbuf.get_width(), pixbuf.get_height()
            ratio = 400 / w
            width, height = w * ratio, h * ratio
            pixbuf_final = pixbuf.scale_simple(
                width, height, GdkPixbuf.InterpType.BILINEAR
            )
            image = Gtk.Image()
            image.set_from_pixbuf(pixbuf_final)
            dialog = Gtk.Window()
            vbox = Gtk.VBox()
            confirm_button = Gtk.Button('Confirm')
            confirm_button.connect('clicked',
                                   self.on_confirm, filename, dialog)
            vbox.add(confirm_button)
            vbox.add(image)
            dialog.add(vbox)
            dialog.show_all()

    def on_confirm(self, host, filename, dialog):

        dialog.destroy()

        from evaluate import heursiticGenerate
        print('filename confirm: ', filename)
        image = cv2.imread(filename)
        heursiticGenerate(image)

    def add_filters(self, dialog):

        filter_text = Gtk.FileFilter()
        filter_text.set_name('Image file')
        filter_text.add_mime_type('image/jpeg')
        dialog.add_filter(filter_text)

        filter_any = Gtk.FileFilter()
        filter_any.set_name('Any files')
        filter_any.add_pattern('*')
        dialog.add_filter(filter_any)


class Application(Gtk.Application):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, application_id="org.example.myapp",
                         flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE,
                         **kwargs)
        self.window = None

        self.add_main_option("test", ord("t"), GLib.OptionFlags.NONE,
                             GLib.OptionArg.NONE, "Command line test", None)

    def do_startup(self):
        Gtk.Application.do_startup(self)

        # self.window = H2L_WINDOW(application=self, title='H2L')
        action = Gio.SimpleAction.new("about", None)
        action.connect("activate", self.on_about)
        self.add_action(action)

        action = Gio.SimpleAction.new("quit", None)
        action.connect("activate", self.on_quit)
        self.add_action(action)

        # builder = Gtk.Builder.new_from_string(MENU_XML, -1)
        # self.set_app_menu(builder.get_object("app-menu"))

    def do_activate(self):
        # We only allow a single window and raise any existing ones
        if not self.window:
            # Windows are associated with the application
            # when the last one is closed the application shuts down
            self.window = H2L_WINDOW(application=self, title="Main Window")
            print('Activate')
        self.window.present()

    def do_command_line(self, command_line):
        options = command_line.get_options_dict()

        if options.contains("test"):
            # This is printed on the main instance
            print("Test argument recieved")

        self.activate()
        return 0

    def on_about(self, action, param):
        about_dialog = Gtk.AboutDialog(transient_for=self.window, modal=True)
        about_dialog.present()

    def on_quit(self, action, param):
        self.quit()


ui = H2L_WINDOW()
ui.connect('delete-event', Gtk.main_quit)
ui.show_all()
Gtk.main()

# if __name__ == '__main__':
#     app = Application()
#     app.run(sys.argv)
