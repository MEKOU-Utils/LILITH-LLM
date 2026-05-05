
cmd:ls *.ppm | foreach { magick $_.Name ($_.BaseName + ".png") }


wasm-build:
wasm-pack build --target web --no-default-features --features wasm-ui
