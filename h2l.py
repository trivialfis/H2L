#!/usr/bin/env python3
#
# Copyright © 2017, 2018 Fis Trivial <ybbs.daans@hotmail.com>
#
# This file is part of H2L.
#
# H2L is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# H2L is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with H2L.  If not, see <http://www.gnu.org/licenses/>.
#

import gi
import cv2
import sys
import argparse
from distutils.version import LooseVersion
from h2l.evaluator import h2l_debug 

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, Gio
from gi.repository import GdkPixbuf

debugger = h2l_debug.h2l_debugger()


class H2L_WINDOW(Gtk.ApplicationWindow):

    def __init__(self, *args, **kwargs):
        # Gtk.Window.__init__(self, title='H2L')
        super().__init__(*args, **kwargs)

        self.vbox = Gtk.VBox(spacing=6)

        button = Gtk.Button('Choose File')
        button.connect('clicked', self.on_file_clicked)
        self.vbox.pack_start(button, True, True, 0)

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

        from h2l.evaluate import heursiticGenerate
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

        action = Gio.SimpleAction.new("about", None)
        action.connect("activate", self.on_about)
        self.add_action(action)

        action = Gio.SimpleAction.new("quit", None)
        action.connect("activate", self.on_quit)
        self.add_action(action)

    def do_activate(self):
        if not self.window:
            self.window = H2L_WINDOW(application=self, title="Main Window")
            print('Activate')
        self.window.present()

    def do_command_line(self, command_line):
        options = command_line.get_options_dict()

        if options.contains("test"):
            print("Test argument recieved")

        self.activate()
        return 0

    def on_about(self, action, param):
        about_dialog = Gtk.AboutDialog(transient_for=self.window, modal=True)
        about_dialog.present()

    def on_quit(self, action, param):
        self.quit()


def gui():
    ui = H2L_WINDOW()
    ui.connect('delete-event', Gtk.main_quit)
    ui.show_all()
    Gtk.main()


def cli():
    raise NotImplementedError('Cli not implemented.')


def _require_version(name, version):
    try:
        local_version = __import__(name).__version__
    except ImportError as e:
        print('Import', name, 'failed, make sure you have it installed')
        raise e
    if LooseVersion(local_version) < LooseVersion(version):
        raise ValueError(
            'Local version:', local_version, 'Required version: ', version
        )


def check_modules():
    try:
        from h2l.evaluate import heursiticGenerate
    except ModuleNotFoundError as e:
        print('Self import failed', file=sys.stderr)
        raise e
    _require_version("numpy", "1.12.1")
    _require_version("skimage", "0.13.1")
    _require_version("cv2", "3.3.0.10")
    _require_version("keras", "2.1.2")
    _require_version("tensorflow", "1.3.1")
    debugger.display("All modules checked, we are safe.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset for training.')
    parser.add_argument('--algorithm', type=str, help='Algorithm used')
    parser.add_argument('--nogui', type=bool, help='Run as command.')
    args = parser.parse_args()
    if args.dataset:
        from h2l import train
    else:
        if args.nogui:
            cli()
        else:
            gui()


if __name__ == '__main__':
    check_modules()
    try:
        parse_args()
    except KeyboardInterrupt:
        print('\nExit')
