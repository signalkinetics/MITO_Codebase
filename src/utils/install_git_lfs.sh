# Script from: https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab
mkdir git_lfs
cd git_lfs
wget https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz
tar xvf git-lfs-linux-amd64-v3.6.1.tar.gz
cd git-lfs-3.6.1/
chmod +x install.sh
sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh
mkdir -p ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
./install.sh
git-lfs --version