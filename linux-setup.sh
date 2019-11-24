# make sure we're in home
cd ~

# set up conda
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
chmod +x Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh
source .bashrc

# install tensorflow 2.0
pip install tf-nightly

# install gym
pip install gym
pip install gym[atari]

# install python-opengl
sudo apt-get install python-opengl

# make sure we have emacs
sudo apt-get install emacs
