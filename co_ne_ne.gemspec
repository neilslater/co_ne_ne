# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'co_ne_ne/version'

Gem::Specification.new do |spec|
  spec.name          = "co_ne_ne"
  spec.version       = RuNeNe::VERSION
  spec.authors       = ["Neil Slater"]
  spec.email         = ["slobo777@gmail.com"]
  spec.description   = %q{Ru(by) Ne(ural) Ne(tworks)}
  spec.summary       = %q{Ru(by) Ne(ural) Ne(tworks)}
  spec.homepage      = "https://github.com/neilslater/co_ne_ne"
  spec.license       = "MIT"

  spec.add_dependency "narray", ">= 0.6.0.8"

  spec.add_development_dependency "yard", ">= 0.8.7.2"
  spec.add_development_dependency "bundler", ">= 1.3"
  spec.add_development_dependency "rspec", ">= 2.13.0"
  spec.add_development_dependency "mocha", ">= 0.14.0"
  spec.add_development_dependency "rake", ">= 1.9.1"
  spec.add_development_dependency "rake-compiler", ">= 0.8.3"
  spec.add_development_dependency "coveralls", ">= 0.6.7"

  spec.files         = `git ls-files`.split($/)
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.test_files    = spec.files.grep(%r{^(test|spec|features)/})
  spec.extensions    = spec.files.grep(%r{/extconf\.rb$})
  spec.require_paths = ["lib"]
end
