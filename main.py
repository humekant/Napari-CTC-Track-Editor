import napari
from src.widgets_cn import CTCEditorWidget

# from src.widgets_en import CTCEditorWidget


def main():
    viewer = napari.Viewer()
    editor_widget = CTCEditorWidget(viewer)
    viewer.window.add_dock_widget(editor_widget, name="CTC Editor", area="right")
    napari.run()


if __name__ == "__main__":
    main()
