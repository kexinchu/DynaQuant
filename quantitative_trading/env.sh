#/bin/bash

#
python3 -m pip install yfinance pandas numpy backtrader

# instll ta
# firstly, install required C lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -zxvf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

python3 -m pip install TA-Lib ta