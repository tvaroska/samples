"""

    Sample to show Gemini with caching to execute agentic reflection
    User upload PDF - gemini create summarization

"""

import os
import uuid
import flet as ft

from datetime import timedelta

from google.cloud import storage

from vertexai.generative_models import GenerativeModel, Part

BUCKET = "btvaroska-llm"
PREFIX = "tmp"

def main(page: ft.Page):

    gcs = storage.Client()
    model = GenerativeModel(model_name="gemini-1.5-pro-001")

    def get_file(e: ft.FilePickerResultEvent):
        if e.files:
            f = e.files[0]
            # TODO: upload file to GCS
            bucket = gcs.bucket(BUCKET)
            dir = str(uuid.uuid4())[:8]
            fname = f'{PREFIX}/{dir}/{f.name}'
            blob = bucket.blob(fname)
            blob_uri = f'gs://{BUCKET}/{fname}'

            signed_url = blob.generate_signed_url(
                version="v4",
                # This URL is valid for 15 minutes
                expiration=timedelta(minutes=15),
                method="PUT",
                content_type="application/octet-stream"
            )

            upload_file = ft.FilePickerUploadFile(
                f.name,
                upload_url=signed_url,
            )

            pick_file_dialog.upload([upload_file])
            
            text_part = Part.from_text('Summarize content of attached paper into three paragraphs')
            pdf_part = Part.from_uri(blob_uri, 'application/pdf')
            response = model.generate_content(contents=[text_part, pdf_part])
            selected_files.value = response.candidates[0].content.parts[0].text
        else:
            selected_files.value = "Cancelled!"
        selected_files.update()

    page.title = "Gemini Reflection sample"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    # example for generating page theme colors based on the seed color
    page.theme = ft.Theme(color_scheme_seed="blue")

    pick_file_dialog = ft.FilePicker(on_result=get_file)
    pick_file_button = ft.IconButton(
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_file_dialog.pick_files(
                        allow_multiple=True
                    ),
                )

    selected_files = ft.Text()
    page.overlay.append(pick_file_dialog)

    page.add(
        ft.AppBar(
            title=ft.Text("Gemini Reflection samples"),
            center_title=True,
            actions=[pick_file_button]
        ),
        ft.Row(
            [
                selected_files,
            ]
        )
    )

os.environ['FLET_SECRET_KEY'] = 'SECRET_KEY'
ft.app(main, view=ft.AppView.WEB_BROWSER)
