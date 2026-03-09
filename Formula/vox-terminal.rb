class VoxTerminal < Formula
  include Language::Python::Virtualenv

  desc "Voice-powered coding assistant for IDEs and terminals"
  homepage "https://github.com/jad/vox-terminal"
  url "https://github.com/jad/vox-terminal/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER"
  license "MIT"

  depends_on "python@3.12"
  depends_on "portaudio"

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      Set your Anthropic API key before using vox-terminal:

        export VOX_TERMINAL_LLM__API_KEY="your-key"

      Add it to ~/.zshrc to persist across sessions.

      Quick start from any project:

        cd /path/to/your/project
        vox-terminal start .
    EOS
  end

  test do
    assert_match "Usage", shell_output("#{bin}/vox-terminal --help")
  end
end
