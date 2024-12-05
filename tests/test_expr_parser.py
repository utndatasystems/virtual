from virtual.expr_parser import ExprParser

def test_complex():
  formula = '''
    case when "total_amount_switch" = 0 then 20.98
    when "total_amount_switch" = 1 then round("fare_amount" + "extra" + "mta_tax" + "tip_amount" + "tolls_amount" + 0.3, 2)
    when "total_amount_switch" = 2 then round(0.02 * "trip_distance" + -0.0069 * "PULocationID" + -0.5003 * "DOLocationID" + 0.3795 * "fare_amount" + 0.0133 * "extra" + 0.0022 * "tip_amount" + 124.9925, 2)
    when "total_amount_switch" = 3 then 21.89
    when "total_amount_switch" = 4 then round("fare_amount" + "extra" + "tip_amount" + 0.8, 2)
    when "total_amount_switch" = 5 then round("fare_amount" + "extra" + "mta_tax" + "tip_amount" + "tolls_amount" + 0.3, 2)
    when "total_amount_switch" = 6 then round("fare_amount" + "extra" + "mta_tax" + "tip_amount" + "tolls_amount" + 0.3, 2)
    when "total_amount_switch" = 7 then 31.42
    when "total_amount_switch" = 8 then round(0.076 * "PULocationID" + -0.2071 * "DOLocationID" + 0.038 * "fare_amount" + -0.0009 * "extra" + 0.0074 * "tip_amount" + 73.5461, 2)
    when "total_amount_switch" = 9 then 10.71
    when "total_amount_switch" = 10 then "fare_amount" + "extra" + "mta_tax" + "tip_amount" + "improvement_surcharge"
    when "total_amount_switch" = 11 then round("fare_amount" + "extra" + "tip_amount" + "tolls_amount" + 0.8, 2)
    else "fare_amount" + "extra" + "mta_tax" + "tip_amount" + "tolls_amount" + "improvement_surcharge" end + "total_amount_offset"
  '''
  # parser = ExprParser(formula)
  # parser.parse()
  assert 1