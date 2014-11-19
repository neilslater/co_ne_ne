require "bundler/gem_tasks"
require "rspec/core/rake_task"
require 'rake/extensiontask'

desc "RuNeNe unit tests"
RSpec::Core::RakeTask.new(:test) do |t|
  t.pattern = "spec/*_spec.rb"
  t.verbose = true
end

gemspec = Gem::Specification.load('co_ne_ne.gemspec')
Rake::ExtensionTask.new do |ext|
  ext.name = 'co_ne_ne'
  ext.source_pattern = "*.{c,h}"
  ext.ext_dir = 'ext/co_ne_ne'
  ext.lib_dir = 'lib/co_ne_ne'
  ext.gem_spec = gemspec
end

task :default => [:compile, :test]
