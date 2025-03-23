# Jinja2ライブラリを取り込みます
from jinja2 import Environment, FileSystemLoader
# bracket_expansionはサードパーティーのライブラリです
# pipを使ってあらかじめインストールする必要があります
from bracket_expansion import bracket_expansion
# テンプレートの環境をオブジェクトとして宣言します
ENV = Environment(loader=FileSystemLoader('.'))
# ENVオブジェクトを宣言した後に、フィルターを追加します。テンプレートから文字列を出力する際に
# ここで指定したbracket_expansion関数が実行されます
ENV.filters['bracket_expansion'] = bracket_expansion
template = ENV.get_template("template_2.j2")
# bracket_expansion関数に、文字列のパターンを渡します。変数名はiface_patternとします
print(template.render(iface_pattern='GigabitEthernet0/[0-3]'))
